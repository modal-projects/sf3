from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from collections import deque
from itertools import combinations, product
from pathlib import Path
from typing import Any

import modal

from src.env import EnvironmentConfig, create_environment
from src.eval.video import MatchVideos
from src.serve import MODELS, POLICY_MODEL_KEY
from src.utils import (
    CHARACTER_MAPPING,
    CONTAINER_REGION,
    MAX_CPU_DIFFICULTY,
    MINUTES,
    PARTICIPANT_LABELS,
    RECENT_MOVE_LIMIT,
    ROUTING_REGION,
    FrameEncoder,
    create_gameplay_image,
    generate_move,
    model_reaction_frames,
    player_state,
)

app = modal.App("sf3-eval")
for model in MODELS.values():
    app.include(model.app)

cache_volume = modal.Volume.from_name("sf3-llm-eval-cache", create_if_missing=True)
eval_image = create_gameplay_image(extra_python_packages=("matplotlib==3.11.1",))

with eval_image.imports():
    from src.eval.plot import write_tournament_plots


ROSTER = tuple(CHARACTER_MAPPING.values())
CHARACTERS = tuple(combinations(ROSTER, 2))
CPU_PLAYERS = {f"cpu:{level}" for level in range(1, MAX_CPU_DIFFICULTY + 1)}
DEFAULT_PLAYERS = (*MODELS, f"cpu:{MAX_CPU_DIFFICULTY}")
K_FACTOR, INITIAL_RATING = 16.0, 1200.0

MATCH_ENVIRONMENT_ATTEMPTS = 3
MATCH_RETRIES = 1
MATCH_DEADLINE = 20 * MINUTES
ARCADE_DEADLINE = 60 * MINUTES
MATCH_TIMEOUT = MATCH_ENVIRONMENT_ATTEMPTS * ARCADE_DEADLINE + 5 * MINUTES
ORCHESTRATE_TIMEOUT = 4 * MATCH_TIMEOUT
CANCELLATION_DRAIN_TIMEOUT = 5
RETRYABLE_MAME_ENVIRONMENT_ERRORS = (
    "Failed to register MAME resources!",
    "Failed to open pipe",
    "Timed out locking characters/super arts",
    "Timed out waiting for 1-player character select",
    "Timed out waiting for 2-player character select",
    "Timed out waiting for fight start",
    "Timed out waiting for the post-KO black frame",
)


def validate_players(players: tuple[str, ...]) -> tuple[str, ...]:
    unknown = [
        player
        for player in players
        if player not in MODELS
        and player not in CPU_PLAYERS
        and not player.startswith("/")
    ]
    if unknown:
        raise ValueError(f"Unknown players: {unknown}")
    if len(players) < 2 or len(set(players)) != len(players):
        raise ValueError(f"Need at least two unique players: {players}")
    if len(CPU_PLAYERS.intersection(players)) > 1:
        raise ValueError(f"At most one CPU player is allowed: {players}")
    # the CPU can only play P2, so it goes last in every pair
    return tuple(sorted(players, key=lambda player: player in CPU_PLAYERS))


def valid_match_outcome(outcome: Any, job: dict[str, Any]) -> bool:
    return (
        isinstance(outcome, dict)
        and outcome.get("match_id") == job["match_id"]
        and outcome.get("pair") == job["pair"]
        and outcome.get("match_idx") == job["match_idx"]
    )


async def call_environment(env: Any, operation: Any, *args: Any) -> Any:
    task = asyncio.create_task(asyncio.to_thread(operation, *args))
    try:
        return await asyncio.shield(task)
    except BaseException:
        env.request_stop()
        try:
            async with asyncio.timeout(CANCELLATION_DRAIN_TIMEOUT):
                await asyncio.shield(task)
        except BaseException:
            pass
        raise


async def close_environment(env: Any) -> None:
    env.request_stop()
    try:
        async with asyncio.timeout(CANCELLATION_DRAIN_TIMEOUT):
            await asyncio.to_thread(env.close)
    except TimeoutError:
        print("Timed out closing MAME; abandoning the worker")
    except Exception as exc:
        print(f"Could not close MAME: {type(exc).__name__}: {exc}")


async def create_environment_async(config: EnvironmentConfig) -> Any:
    task = asyncio.create_task(asyncio.to_thread(create_environment, config))
    try:
        return await asyncio.shield(task)
    except BaseException:
        try:
            async with asyncio.timeout(CANCELLATION_DRAIN_TIMEOUT):
                await close_environment(await asyncio.shield(task))
        except BaseException:
            pass
        raise


async def play_game(
    env: Any,
    player1: str,
    player2: str,
    chats: dict[str, Any],
    encoder: FrameEncoder,
    initial_reset: tuple[dict[str, Any], dict[str, Any]],
    videos: MatchVideos,
) -> tuple[str, list[float], list[dict[str, Any]]]:
    if player1 in CPU_PLAYERS:
        raise ValueError("CPU tournaments require CPU in P2")
    vs_cpu = player2 in CPU_PLAYERS
    recent = [deque(maxlen=RECENT_MOVE_LIMIT), deque(maxlen=RECENT_MOVE_LIMIT)]
    observation, _ = initial_reset
    hp_diff = 0.0
    p1_round_hp_diffs: list[float] = []
    fight_round_hp_diffs: list[float] = []
    fights: list[dict[str, Any]] = []
    while True:
        identity = env.read_match_identity()
        fighters = [
            player_state(observation, identity["player1"], "P1"),
            player_state(observation, identity["player2"], "P2"),
        ]
        if videos.path is None:
            videos.start([fighter.character for fighter in fighters])
            videos.write(observation["frame"])
        frame_url = encoder.data_url(observation["frame"])
        players = (player1, player2)
        request_seats = range(1 if vs_cpu else 2)

        async def timed_move(seat):
            started_at = time.perf_counter()
            buttons, name = await generate_move(
                chats[players[seat]],
                fighters[seat],
                fighters[1 - seat],
                frame_url,
                recent[seat],
            )
            generation_ms = (time.perf_counter() - started_at) * 1000
            delay = model_reaction_frames(buttons, generation_ms)
            return [0] * delay + buttons, name

        try:
            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(timed_move(seat)) for seat in request_seats]
        except BaseExceptionGroup as group_error:
            raise group_error.exceptions[0] from group_error
        generated = [task.result() for task in tasks]
        buttons = [[0], [0]]
        for seat, (move_buttons, move_name) in zip(request_seats, generated):
            buttons[seat] = move_buttons
            recent[seat].append(move_name)
        n_steps = max(map(len, buttons))
        actions = zip(
            buttons[0] + [0] * (n_steps - len(buttons[0])),
            buttons[1] + [0] * (n_steps - len(buttons[1])),
        )
        frames = [{"agent_0": p1, "agent_1": p2} for p1, p2 in actions]
        observation, reward, terminated, truncated, info = await call_environment(
            env, videos.step, env, frames
        )
        hp_diff += reward

        if info.get("round_done"):
            p1_round_hp_diffs.append(hp_diff)
            fight_round_hp_diffs.append(hp_diff)
            hp_diff = 0.0
            for moves in recent:
                moves.clear()
        if (vs_cpu and info.get("stage_done")) or terminated:
            fight_winner = (
                player1
                if info.get("stage_done") or info.get("winner") == "P1"
                else "draw"
                if info.get("winner") == "draw"
                else player2
            )
            fight = {
                "winner": fight_winner,
                "characters": [fighter.character for fighter in fighters],
                "p1_round_hp_diffs": fight_round_hp_diffs,
            }
            fight["video"] = str(videos.finish().relative_to(Path("/cache")))
            fights.append(fight)
            fight_round_hp_diffs = []
        if not (terminated or truncated):
            continue
        if truncated or not info.get("game_done"):
            raise RuntimeError("Episode ended without normal game completion")
        seat_winner = info.get("winner")
        if seat_winner == "draw":
            return "draw", p1_round_hp_diffs, fights
        if seat_winner not in {"P1", "P2"}:
            raise RuntimeError(f"Environment returned invalid winner: {seat_winner!r}")
        return (
            player1 if seat_winner == "P1" else player2,
            p1_round_hp_diffs,
            fights,
        )


async def play_match(
    job: dict[str, Any],
    chats: dict[str, Any],
    model_boots: list[Any],
) -> dict[str, Any]:
    a, b = job["pair"]
    match_idx = job["match_idx"]
    vs_cpu = b in CPU_PLAYERS
    # alternate seats between LLMs so the P1 side advantage cancels out
    player1, player2 = (a, b) if vs_cpu or match_idx % 2 == 0 else (b, a)
    timeout_seconds = ARCADE_DEADLINE if vs_cpu else MATCH_DEADLINE
    video_prefix = hashlib.sha256(job["match_id"].encode()).hexdigest()[:12]
    videos = MatchVideos(
        Path("/cache") / job["run_id"] / "videos",
        (player1, player2),
        f"{match_idx:03d}-{video_prefix}",
    )
    deadline = asyncio.timeout(timeout_seconds)
    try:
        async with deadline:
            encoder = FrameEncoder()
            env_task = asyncio.create_task(
                create_environment_async(
                    EnvironmentConfig(
                        characters=tuple(job["characters"]),
                        outfits=(1, 1),
                        super_arts=(1, 1),
                        step_ratio=1,
                        vs_cpu=vs_cpu,
                        lock_cpu_opponent=not vs_cpu,
                        cpu_difficulty=int(b.removeprefix("cpu:"))
                        if vs_cpu
                        else MAX_CPU_DIFFICULTY,
                    )
                )
            )
            boot_tasks = [asyncio.create_task(boot()) for boot in model_boots]
            startup_tasks = [env_task, *boot_tasks]
            env = None
            try:
                env = await env_task
                reset_task = asyncio.create_task(call_environment(env, env.reset))
                startup_tasks.append(reset_task)
                await asyncio.gather(reset_task, *boot_tasks)
            except BaseException:
                for task in startup_tasks:
                    task.cancel()
                await asyncio.gather(*startup_tasks, return_exceptions=True)
                if env is not None:
                    await close_environment(env)
                raise
            try:
                winner, p1_round_hp_diffs, fights = await play_game(
                    env,
                    player1,
                    player2,
                    chats,
                    encoder,
                    reset_task.result(),
                    videos,
                )
                if winner != "draw" and winner not in {a, b}:
                    raise RuntimeError(f"Invalid game winner: {winner!r}")
            finally:
                try:
                    await close_environment(env)
                finally:
                    await asyncio.to_thread(videos.close)
    except TimeoutError as exc:
        if not deadline.expired():
            raise
        raise TimeoutError(
            f"{a} vs {b} exceeded {timeout_seconds / MINUTES:.0f} minutes"
        ) from exc
    return {
        "match_id": job["match_id"],
        "match_idx": match_idx,
        "pair": [a, b],
        "player1": player1,
        "winner": winner,
        "p1_round_hp_diffs": p1_round_hp_diffs,
        "fights": fights,
    }


@app.function(
    image=eval_image,
    volumes={"/cache": cache_volume},
    region=CONTAINER_REGION,
    routing_region=ROUTING_REGION,
    timeout=MATCH_TIMEOUT,
    retries=MATCH_RETRIES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
async def execute_match(job: dict[str, Any]) -> dict[str, Any]:
    chats = {}
    model_boots = []
    for player in job["pair"]:
        if player in CPU_PLAYERS:
            continue
        spec = MODELS.get(player, MODELS[POLICY_MODEL_KEY])
        # the policy server loads the latest fine-tuned export when ckpt_path is empty
        ckpt_path = player if player.startswith("/") else spec.version["model"]
        server = spec.server.with_options(gpu=spec.eval_gpu)(ckpt_path=ckpt_path)
        chats[player] = server.chat.remote.aio
        model_boots.append(server.boot.remote.aio)

    for attempt in range(MATCH_ENVIRONMENT_ATTEMPTS):
        try:
            outcome = await play_match(job, chats, model_boots)
            await cache_volume.commit.aio()
            return outcome
        except (OSError, TimeoutError) as exc:
            retryable = any(
                marker in str(exc) for marker in RETRYABLE_MAME_ENVIRONMENT_ERRORS
            )
            if not retryable or attempt + 1 == MATCH_ENVIRONMENT_ATTEMPTS:
                raise
            print(
                f"Retrying {job['match_id']} after MAME environment error "
                f"({attempt + 1}/{MATCH_ENVIRONMENT_ATTEMPTS})"
            )
    raise AssertionError("unreachable")


@app.function(
    image=eval_image,
    volumes={"/cache": cache_volume},
    region=CONTAINER_REGION,
    routing_region=ROUTING_REGION,
    timeout=ORCHESTRATE_TIMEOUT,
)
async def orchestrate(players: tuple[str, ...] = DEFAULT_PLAYERS) -> dict[str, Any]:
    players = validate_players(tuple(players))
    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}-{uuid.uuid4().hex[:8]}"
    run_output_dir = run_id
    jobs = [
        {
            "match_id": f"{run_id}-{a}-{b}-{match_idx}",
            "run_id": run_id,
            "match_idx": match_idx,
            "pair": [a, b],
            "characters": (
                [ROSTER[match_idx % len(ROSTER)], characters[1]]
                if b in CPU_PLAYERS
                else list(characters)
            ),
        }
        for (a, b), (match_idx, characters) in product(
            combinations(players, 2), enumerate(CHARACTERS)
        )
    ]
    config = {
        "run_id": run_id,
        "players": players,
        "characters": list(CHARACTERS),
        "llm_match_format": "one_fight_first_to_two_rounds",
        "cpu_match_format": "full_arcade_ladder",
        "win_rate_unit": "fight_first_to_two_rounds",
        "scheduled_games_per_pair": len(CHARACTERS),
        "models": {
            player: {"model": player}
            if player.startswith("/")
            else MODELS[player].version
            for player in players
            if player not in CPU_PLAYERS
        },
        "output_dir": run_output_dir,
        "schedule": jobs,
    }
    outcomes: list[Any] = [None] * len(jobs)
    completed = failed = 0
    print(f"run_id: {run_id}")
    print(f"players: {players}")
    print(f"n_matches: {len(jobs)}")

    index = 0
    async for outcome in execute_match.map.aio(jobs, return_exceptions=True):
        outcomes[index] = outcome
        if isinstance(outcome, BaseException):
            failed += 1
            print(
                f"Match failed: {jobs[index]['match_id']}: "
                f"{type(outcome).__name__}: {outcome}",
                flush=True,
            )
        else:
            completed += 1
        index += 1
        print(
            f"Progress: completed={completed} failed={failed} "
            f"pending={len(jobs) - completed - failed}",
            flush=True,
        )

    # build report

    games: list[dict[str, Any]] = []
    failures = []
    for job, outcome in zip(jobs, outcomes):
        if isinstance(outcome, BaseException):
            failures.append(f"{job['match_id']}: {type(outcome).__name__}: {outcome}")
        elif not valid_match_outcome(outcome, job):
            failures.append(f"{job['match_id']}: worker returned a mismatched result")
        else:
            games.append(outcome)
    complete = not failures and len(games) == len(jobs)
    matches: list[dict[str, Any]] = [
        {
            **{key: value for key, value in game.items() if key != "fights"},
            **fight,
            "game_id": game["match_id"],
            "match_id": f"{game['match_id']}/fight-{fight_idx}",
            "fight_idx": fight_idx,
        }
        for game in games
        for fight_idx, fight in enumerate(game["fights"])
    ]
    report = {
        "complete": complete,
        "config": config,
        "matches": matches,
        "games": games,
        "arcade_runs": [
            {
                "game_id": game["match_id"],
                "player": game["player1"],
                "cpu": game["pair"][1],
                "cleared": game["winner"] == game["player1"],
                "fights": len(game["fights"]),
            }
            for game in games
            if game["pair"][1] in CPU_PLAYERS
        ],
        "failures": failures,
    }
    player_index = {player: index for index, player in enumerate(config["players"])}
    ratings = dict.fromkeys(config["players"], INITIAL_RATING)
    for match in sorted(
        matches,
        key=lambda item: (
            item["match_idx"],
            player_index[item["pair"][0]],
            player_index[item["pair"][1]],
            item["fight_idx"],
        ),
    ):
        a, b = match["pair"]
        score_a = 0.5 if match["winner"] == "draw" else float(match["winner"] == a)
        expected_a = 1 / (1 + 10 ** ((ratings[b] - ratings[a]) / 400))
        ratings[a] += K_FACTOR * (score_a - expected_a)
        ratings[b] += K_FACTOR * (expected_a - score_a)
    pair_results: dict[tuple[str, str], dict[str, Any]] = {}
    for match in matches:
        a, b = match["pair"]
        summary = pair_results.setdefault(
            (a, b),
            {
                "pair": [a, b],
                "matches": 0,
                "wins": {a: 0, b: 0},
                "draws": 0,
                "rounds": 0,
                "round_wins": {a: 0, b: 0},
                "hp_diff_sum": 0.0,
            },
        )
        summary["matches"] += 1
        if match["winner"] == "draw":
            summary["draws"] += 1
        else:
            summary["wins"][match["winner"]] += 1
        for p1_hp_diff in match["p1_round_hp_diffs"]:
            hp_diff = p1_hp_diff if match["player1"] == a else -p1_hp_diff
            summary["rounds"] += 1
            summary["round_wins"][a] += hp_diff > 0
            summary["round_wins"][b] += hp_diff < 0
            summary["hp_diff_sum"] += hp_diff
    pair_summaries = sorted(
        pair_results.values(),
        key=lambda summary: (
            player_index[summary["pair"][0]],
            player_index[summary["pair"][1]],
        ),
    )
    win_rate_matrix, round_win_rate_matrix = (
        {
            player: {
                opponent: None if player == opponent else 0.0
                for opponent in config["players"]
            }
            for player in config["players"]
        }
        for _ in range(2)
    )
    for summary in pair_summaries:
        a, b = summary["pair"]
        summary["match_win_rates"] = {
            a: summary["wins"][a] / summary["matches"],
            b: summary["wins"][b] / summary["matches"],
        }
        summary["round_win_rates"] = {
            a: summary["round_wins"][a] / summary["rounds"],
            b: summary["round_wins"][b] / summary["rounds"],
        }
        summary["mean_hp_diff"] = summary["hp_diff_sum"] / summary["rounds"]
        win_rate_matrix[a][b] = summary["match_win_rates"][a]
        win_rate_matrix[b][a] = summary["match_win_rates"][b]
        round_win_rate_matrix[a][b] = summary["round_win_rates"][a]
        round_win_rate_matrix[b][a] = summary["round_win_rates"][b]
    report["pair_results"] = pair_summaries
    report["win_rate_matrix"] = win_rate_matrix
    report["round_win_rate_matrix"] = round_win_rate_matrix
    order = sorted(ratings, key=lambda player: (-ratings[player], player_index[player]))
    report["rankings"] = [
        {
            "rank": rank,
            "player": player,
            "label": PARTICIPANT_LABELS.get(player, Path(player).name),
            "elo": round(ratings[player], 4),
        }
        for rank, player in enumerate(order, 1)
    ]

    # save report

    await cache_volume.reload.aio()
    directory = Path("/cache") / run_output_dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("results.json" if report["complete"] else "partial.json")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(path)

    # save plots

    if report["complete"]:
        for plot in write_tournament_plots(report, directory):
            print(f"Saved {plot}")
    print(f"Saved {path}")
    report["volume_path"] = str(path)
    await cache_volume.commit.aio()
    if not report["complete"]:
        raise RuntimeError(
            f"Tournament incomplete ({len(report['failures'])}/{len(jobs)} failed); "
            f"wrote non-authoritative {report['volume_path']}"
        )
    for row in report["rankings"]:
        print(f"{row['rank']}. {row['label']}: {row['elo']:.2f}")
    return report


@app.local_entrypoint()
async def main(players: str = ",".join(DEFAULT_PLAYERS)) -> None:
    participants = validate_players(tuple(players.split(",")))
    call = await orchestrate.spawn.aio(players=participants)
    print(f"call_id={call.object_id}", flush=True)
