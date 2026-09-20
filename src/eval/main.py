from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import deque
from itertools import combinations, product
from pathlib import Path
from typing import Any

import modal

from src.env import EnvironmentConfig, create_environment
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
    player_state,
)

app = modal.App("sf3-eval")
for model in MODELS.values():
    app.include(model.app)

cache_volume = modal.Volume.from_name("sf3-llm-eval-cache", create_if_missing=True)
eval_image = create_gameplay_image(extra_python_packages=("matplotlib==3.11.1",))

with eval_image.imports():
    from src.eval.plot import write_tournament_plots


CHARACTERS = tuple(combinations(CHARACTER_MAPPING.values(), 2))
K_FACTOR, INITIAL_RATING = 16.0, 1200.0
OUTPUT_DIR = "tournaments"

MATCH_ENVIRONMENT_ATTEMPTS = 3
MATCH_RETRIES = 1
MATCH_DEADLINE = 20 * MINUTES
MATCH_TIMEOUT = MATCH_ENVIRONMENT_ATTEMPTS * MATCH_DEADLINE + 5 * MINUTES
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
) -> str:
    if player1 == "cpu":
        raise ValueError("CPU tournaments require CPU in P2")
    vs_cpu = player2 == "cpu"
    recent = [deque(maxlen=RECENT_MOVE_LIMIT), deque(maxlen=RECENT_MOVE_LIMIT)]
    observation, _ = initial_reset
    while True:
        identity = env.read_match_identity()
        fighters = [
            player_state(observation, identity["player1"], "P1"),
            player_state(observation, identity["player2"], "P2"),
        ]
        frame_url = encoder.data_url(observation["frame"])
        players = (player1, player2)
        request_seats = range(1 if vs_cpu else 2)
        try:
            async with asyncio.TaskGroup() as group:
                tasks = [
                    group.create_task(
                        generate_move(
                            chats[players[seat]],
                            fighters[seat],
                            fighters[1 - seat],
                            frame_url,
                            recent[seat],
                        )
                    )
                    for seat in request_seats
                ]
        except BaseExceptionGroup as group_error:
            # match retry and deadline handling both match on the concrete
            # exception type, which a group would hide
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

        terminated = truncated = False
        info: dict[str, Any] = {}
        for p1_button, p2_button in actions:
            observation, _, terminated, truncated, info = await call_environment(
                env,
                env.step,
                {"agent_0": p1_button, "agent_1": p2_button},
            )
            if terminated or truncated or info.get("round_done"):
                break

        if info.get("round_done"):
            for moves in recent:
                moves.clear()
        if not (terminated or truncated):
            continue
        if truncated or not info.get("game_done"):
            raise RuntimeError("Episode ended without normal game completion")
        seat_winner = info.get("winner")
        if seat_winner == "draw":
            return "draw"
        if seat_winner not in {"P1", "P2"}:
            raise RuntimeError(f"Environment returned invalid winner: {seat_winner!r}")
        return player1 if seat_winner == "P1" else player2


async def play_match(
    job: dict[str, Any],
    chats: dict[str, Any],
    model_boots: list[Any],
) -> dict[str, Any]:
    a, b = job["pair"]
    match_idx = job["match_idx"]
    vs_cpu = b == "cpu"
    # alternate seats between LLMs so the P1 side advantage cancels out
    player1, player2 = (a, b) if vs_cpu or match_idx % 2 == 0 else (b, a)
    deadline = asyncio.timeout(MATCH_DEADLINE)
    try:
        async with deadline:
            encoder = FrameEncoder()
            env_task = asyncio.create_task(
                create_environment_async(
                    EnvironmentConfig(
                        characters=tuple(job["characters"]),
                        outfits=(1, 1),
                        super_arts=(1, 1),
                        step_ratio=6,
                        vs_cpu=vs_cpu,
                        cpu_difficulty=MAX_CPU_DIFFICULTY,
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
                winner = await play_game(
                    env,
                    player1,
                    player2,
                    chats,
                    encoder,
                    reset_task.result(),
                )
                if winner != "draw" and winner not in {a, b}:
                    raise RuntimeError(f"Invalid game winner: {winner!r}")
            finally:
                await close_environment(env)
    except TimeoutError as exc:
        if not deadline.expired():
            raise
        raise TimeoutError(
            f"{a} vs {b} exceeded {MATCH_DEADLINE / MINUTES:.0f} minutes"
        ) from exc
    return {
        "match_id": job["match_id"],
        "match_idx": match_idx,
        "pair": [a, b],
        "mode": "cpu_tournament" if vs_cpu else "ft2_rounds",
        "player1": player1,
        "winner": winner,
    }


@app.function(
    image=eval_image,
    region=CONTAINER_REGION,
    routing_region=ROUTING_REGION,
    timeout=MATCH_TIMEOUT,
    retries=MATCH_RETRIES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
async def execute_match(
    job: dict[str, Any],
    checkpoint_name: str,
    base: bool = False,
    ckpt_path: str = "",
) -> dict[str, Any]:
    chats = {}
    model_boots = []
    for player in job["pair"]:
        if player == "cpu":
            continue
        spec = MODELS[player]
        if player == POLICY_MODEL_KEY:
            player_ckpt_path = spec.version["model"] if base else ckpt_path
        else:
            player_ckpt_path = ""
        server = spec.server.with_options(gpu=spec.eval_gpu)(ckpt_path=player_ckpt_path)
        chats[player] = server.chat.remote.aio
        model_boots.append(server.boot.remote.aio)

    for attempt in range(MATCH_ENVIRONMENT_ATTEMPTS):
        try:
            result = await play_match(job, chats, model_boots)
            checkpoint = modal.Dict.from_name(checkpoint_name, create_if_missing=True)
            await checkpoint.put.aio(f"match:{job['match_id']}", result)
            return result
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
async def orchestrate(base: bool = False, ckpt_path: str = "") -> dict[str, Any]:
    players = list((*MODELS, "cpu"))
    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}-{uuid.uuid4().hex[:8]}"
    run_output_dir = f"{OUTPUT_DIR}/{run_id}"
    checkpoint_name = f"sf3-eval-{run_id}"
    jobs = [
        {
            "match_id": f"{run_id}-{a}-{b}-{match_idx}",
            "match_idx": match_idx,
            "pair": [a, b],
            "characters": list(characters),
        }
        for (a, b), (match_idx, characters) in product(
            combinations(players, 2), enumerate(CHARACTERS)
        )
    ]
    ckpt_selection = "base" if base else (ckpt_path if ckpt_path else "latest")
    config = {
        "run_id": run_id,
        "players": players,
        "characters": list(CHARACTERS),
        "cpu_difficulty": MAX_CPU_DIFFICULTY,
        "llm_match_format": "one_fight_first_to_two_rounds",
        "matches_per_pair": len(CHARACTERS),
        "models": {
            player: MODELS[player].version for player in players if player != "cpu"
        },
        "ckpt_selection": ckpt_selection,
        "output_dir": run_output_dir,
        "schedule": jobs,
    }
    checkpoint = modal.Dict.from_name(checkpoint_name, create_if_missing=True)
    outcomes: list[Any] = [None] * len(jobs)
    completed = failed = 0
    print(f"run_id: {run_id}")
    print(f"ckpt_selection: {ckpt_selection}")
    print(f"n_matches: {len(jobs)}")

    index = 0
    async for outcome in execute_match.map.aio(
        jobs,
        kwargs={
            "checkpoint_name": checkpoint_name,
            "base": base,
            "ckpt_path": ckpt_path,
        },
        return_exceptions=True,
    ):
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

    # worker can checkpoint its result and still fail on the way home so double-check

    failed_indexes = [
        index
        for index, outcome in enumerate(outcomes)
        if isinstance(outcome, BaseException)
    ]
    if failed_indexes:
        reconciled = await asyncio.gather(
            *(
                checkpoint.get.aio(f"match:{jobs[index]['match_id']}")
                for index in failed_indexes
            )
        )
        for index, outcome in zip(failed_indexes, reconciled):
            if valid_match_outcome(outcome, jobs[index]):
                outcomes[index] = outcome

    # build report

    matches = []
    failures = []
    for job, outcome in zip(jobs, outcomes):
        if isinstance(outcome, BaseException):
            failures.append(f"{job['match_id']}: {type(outcome).__name__}: {outcome}")
        elif not valid_match_outcome(outcome, job):
            failures.append(f"{job['match_id']}: worker returned a mismatched result")
        else:
            matches.append(outcome)
    complete = not failures and len(matches) == len(jobs)
    report = {
        "complete": complete,
        "config": config,
        "matches": matches,
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
            {"pair": [a, b], "matches": 0, "wins": {a: 0, b: 0}, "draws": 0},
        )
        summary["matches"] += 1
        if match["winner"] == "draw":
            summary["draws"] += 1
        else:
            summary["wins"][match["winner"]] += 1
    pair_summaries = sorted(
        pair_results.values(),
        key=lambda summary: (
            player_index[summary["pair"][0]],
            player_index[summary["pair"][1]],
        ),
    )
    win_rate_matrix = {
        player: {
            opponent: None if player == opponent else 0.0
            for opponent in config["players"]
        }
        for player in config["players"]
    }
    for summary in pair_summaries:
        a, b = summary["pair"]
        summary["match_win_rates"] = {
            a: summary["wins"][a] / summary["matches"],
            b: summary["wins"][b] / summary["matches"],
        }
        win_rate_matrix[a][b] = summary["match_win_rates"][a]
        win_rate_matrix[b][a] = summary["match_win_rates"][b]
    report["pair_results"] = pair_summaries
    report["win_rate_matrix"] = win_rate_matrix
    order = sorted(ratings, key=lambda player: (-ratings[player], player_index[player]))
    report["rankings"] = [
        {
            "rank": rank,
            "player": player,
            "label": PARTICIPANT_LABELS[player],
            "elo": round(ratings[player], 4),
        }
        for rank, player in enumerate(order, 1)
    ]
    # save report

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
async def main(base: bool = False, ckpt_path: str = "") -> None:
    call = await orchestrate.spawn.aio(base=base, ckpt_path=ckpt_path)
    print(f"call_id={call.object_id}", flush=True)
