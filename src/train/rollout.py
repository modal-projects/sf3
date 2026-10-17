import logging
import random
import socket
import statistics
import traceback
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from types import SimpleNamespace

import modal
from aiohttp import web
from modal_dojo import Qwen3_VL_8B

from src.env import EnvironmentConfig
from src.train.worker import FRAME_KEY, run_arcade_sandbox
from src.utils import (
    CHARACTER_MAPPING,
    HEALTH_MAX,
    MAX_CPU_DIFFICULTY,
    PlayerState,
    create_messages,
    repeat_exclusion_probability,
)

MODEL = Qwen3_VL_8B()
ROSTER = tuple(CHARACTER_MAPPING.values())
OUTFIT = SUPER_ART = 1
IMAGE_PAD = "<|image_pad|>"

DISCOUNT_HALF_LIFE_FRAMES = 60 * 60
DAMAGE_REWARD_WEIGHT = 0.1
INFERENCE_LATENCY_MS = 200.0  # estimate of deployed inference latency, independent of training batch latency.
ROUND_WIN_BONUS = 1.0
INVALID_MOVE_PENALTY = 20 / HEALTH_MAX
MIN_CPU_RESULTS = 10

_results: defaultdict[int, deque] = defaultdict(lambda: deque(maxlen=50))
_fight_counts: defaultdict[int, int] = defaultdict(int)
_reward_baselines: dict[int, float] = {}
_image_tokens_by_frame_size: dict[tuple[int, int], int] = {}


async def _move(args, sample, sampling_params, fighters, frame, frame_size, recent):
    from slime.rollout.sglang_rollout import GenerateState
    from slime.utils.http_utils import post
    from slime.utils.types import Sample

    state = GenerateState(args)
    if state.aborted:
        raise RuntimeError("rollout aborted")
    probability = args.sf3_repeat_exclusion_probability
    exclude_recent = random.random() < probability
    messages, available = create_messages(
        fighters[1], fighters[0], [frame], recent if exclude_recent else None
    )
    prompt_text = state.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    count = _image_tokens_by_frame_size.get(frame_size)
    if count is None:
        prompt_ids = state.processor(
            text=prompt_text, images=[_image(frame)], return_tensors="pt"
        )["input_ids"][0].tolist()
        _image_tokens_by_frame_size[frame_size] = prompt_ids.count(
            state.tokenizer.convert_tokens_to_ids(IMAGE_PAD)
        )
    else:
        prompt_ids = state.tokenizer(
            prompt_text.replace(IMAGE_PAD, IMAGE_PAD * count, 1),
            add_special_tokens=False,
        )["input_ids"]
    move = Sample(
        group_index=sample.group_index,
        index=sample.index,
        rollout_id=sample.index,
        prompt=prompt_text,
        tokens=prompt_ids,
        multimodal_train_inputs={FRAME_KEY: frame},
    )
    move.metadata.update(
        repeat_exclusion_active=exclude_recent,
        repeat_exclusion_probability=probability,
    )
    output = await post(
        f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate",
        {
            "text": prompt_text,
            "image_data": [frame],
            "sampling_params": sampling_params,
            "return_logprob": True,
        },
    )
    if output["meta_info"]["prompt_tokens"] != len(prompt_ids):
        raise RuntimeError("SGLang prompt tokens differ from local tokenization")
    logprobs = output["meta_info"]["output_token_logprobs"]
    move.append_response_tokens(
        args,
        tokens=[item[1] for item in logprobs],
        log_probs=[item[0] for item in logprobs],
        trainable=True,
        meta_info=output["meta_info"],
        text=output["text"],
    )
    if move.status == Sample.Status.ABORTED:
        raise RuntimeError("SGLang aborted a move")
    return move, available


def _win_rate(results):
    return (sum(results) + 1) / (len(results) + 2)


def _sample_cpu_difficulty():
    # PFSP-style: https://www.nature.com/articles/s41586-019-1724-z
    hardest_won = max(
        (
            d
            for d in range(1, MAX_CPU_DIFFICULTY + 1)
            if len(r := _results[d]) >= MIN_CPU_RESULTS and _win_rate(r) >= 0.5
        ),
        default=0,
    )
    levels = range(1, min(hardest_won + 1, MAX_CPU_DIFFICULTY) + 1)
    rates = [_win_rate(_results[d]) for d in levels]
    return random.choices(levels, [p * (1 - p) for p in rates])[0]


@asynccontextmanager
async def _inference_endpoint(args):
    async def generate(request):
        payload = await request.json()
        move, available = await _move(
            args,
            SimpleNamespace(**payload["sample"]),
            payload["sampling_params"],
            [PlayerState(**fighter) for fighter in payload["fighters"]],
            payload["frame"],
            tuple(payload["frame_size"]),
            payload["recent"],
        )
        data = move.to_dict()
        # The worker already holds the image, so remove to avoid sending it over HTTP.
        data.pop("multimodal_train_inputs", None)
        for key, value in data.items():
            if hasattr(value, "tolist"):
                data[key] = value.tolist()
        return web.json_response(
            {
                "sample": data,
                "name": MODEL.parse_response(move.response).content.strip(),
                "available": available,
            }
        )

    app = web.Application()
    app.router.add_post("/move", generate)
    runner = web.AppRunner(app, access_log=None)
    with socket.socket() as listener:
        await runner.setup()
        try:
            listener.bind(("0.0.0.0", 0))
            listener.listen()
            await web.SockSite(runner, listener).start()
            async with modal.forward.aio(listener.getsockname()[1]) as tunnel:
                yield tunnel.url
        finally:
            await runner.cleanup()


def sf3_rollout(args, rollout_id, data_source, evaluation=False):
    from slime.rollout.sglang_rollout import generate_rollout_async
    from slime.utils.async_utils import run

    if evaluation:
        raise ValueError(
            "SF3 training uses fight metrics instead of evaluation rollouts"
        )

    state = data_source.metadata.get("sf3_state")
    if state and not any(_results.values()):
        _results.update(
            {
                level: deque(results, maxlen=50)
                for level, results in state["results"].items()
            }
        )
        _fight_counts.update(state["fight_counts"])
        _reward_baselines.update(state["reward_baselines"])
    anneal_start = data_source.metadata.setdefault("sf3_anneal_start", rollout_id)

    async def collect():
        async with _inference_endpoint(args) as endpoint:
            args.sf3_inference_url = endpoint
            args.sf3_repeat_exclusion_probability = repeat_exclusion_probability(
                rollout_id - anneal_start
            )
            try:
                output, aborted = await generate_rollout_async(
                    args, rollout_id, data_source.get_samples
                )
                if aborted:
                    data_source.add_samples(aborted)
                data_source.metadata["sf3_state"] = {
                    "results": {
                        level: list(results) for level, results in _results.items()
                    },
                    "fight_counts": dict(_fight_counts),
                    "reward_baselines": _reward_baselines,
                }
                return output
            finally:
                del args.sf3_inference_url
                del args.sf3_repeat_exclusion_probability

    return run(collect())


def _assign_rewards(moves, invalid, damage, durations, winner):
    outcome = ROUND_WIN_BONUS * {"P1": 1, "P2": -1, "draw": 0}[winner]
    damage_return = 0.0
    for move, bad, hp_diff, frames in reversed(
        list(zip(moves, invalid, damage, durations, strict=True))
    ):
        discount = 0.5 ** (frames / DISCOUNT_HALF_LIFE_FRAMES)
        outcome *= discount
        damage_return = hp_diff / HEALTH_MAX + discount * damage_return
        move.reward = (
            outcome
            + DAMAGE_REWARD_WEIGHT * max(-1.0, min(1.0, damage_return))
            - INVALID_MOVE_PENALTY * bad
        )
        move.metadata.update(
            round_winner=winner,
            damage=hp_diff,
            decision_frames=frames,
            invalid_move=bool(bad),
        )


async def _play_arcade(args, sample, sampling_params):
    from slime.utils.types import Sample

    difficulty = _sample_cpu_difficulty()
    characters = random.sample(ROSTER, 2)
    config = EnvironmentConfig(
        characters=(characters[0], characters[1]),
        outfits=(OUTFIT, OUTFIT),
        super_arts=(SUPER_ART, SUPER_ART),
        step_ratio=1,
        vs_cpu=True,
        cpu_difficulty=difficulty,
        lock_cpu_opponent=False,
    )
    result = await run_arcade_sandbox(
        config,
        args.sf3_inference_url,
        {"index": sample.index, "group_index": sample.group_index},
        sampling_params,
        INFERENCE_LATENCY_MS,
    )
    all_moves = []
    for round_data in result["rounds"]:
        moves = [Sample.from_dict(move) for move in round_data["moves"]]
        for move in moves:
            move.metadata["cpu_difficulty"] = difficulty
            move.metadata["sandbox_id"] = result["sandbox_id"]
        _assign_rewards(
            moves,
            round_data["invalid"],
            round_data["damage"],
            round_data["durations"],
            round_data["winner"],
        )
        all_moves.extend(moves)
    _results[difficulty].extend(result["results"])
    _fight_counts[difficulty] += len(result["results"])
    # slime's static batching needs every step's sample count divisible by
    # DP * micro_batch_size.
    gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
    align = gpus * args.micro_batch_size
    keep = random.sample(all_moves, len(all_moves) // align * align)
    if not keep:
        raise RuntimeError("arcade run too short to align with DP * micro_batch_size")
    keep[0].metadata["image"] = keep[0].multimodal_train_inputs[FRAME_KEY]
    return keep


async def sf3_generate(args, sample, sampling_params):
    from slime.utils.types import Sample

    try:
        return await _play_arcade(args, sample, sampling_params)
    except Exception:
        traceback.print_exc()
        sample.status = Sample.Status.ABORTED
        return [sample]


def sf3_reward_post_process(args, samples):
    rewards = [sample.reward for sample in samples]
    groups = defaultdict(list)
    for i, sample in enumerate(samples):
        groups[sample.metadata["cpu_difficulty"]].append(i)
    advantages = list(rewards)
    for difficulty, members in groups.items():
        baseline = _reward_baselines.get(difficulty, 0.0)
        for i in members:
            advantages[i] -= baseline
        mean = statistics.fmean(rewards[i] for i in members)
        _reward_baselines[difficulty] = 0.9 * baseline + 0.1 * mean
    return rewards, advantages


def sf3_log_rollout(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    from slime.utils import logging_utils
    from slime.utils.metric_utils import compute_rollout_step

    metrics = {
        "rollout/step": compute_rollout_step(args, rollout_id),
        "sf3/invalid_move_rate": statistics.fmean(
            sample.metadata["invalid_move"] for sample in samples
        ),
        "sf3/repeat_exclusion_probability": statistics.fmean(
            sample.metadata["repeat_exclusion_probability"] for sample in samples
        ),
    }
    for difficulty, results in _results.items():
        if results:
            prefix = f"opponent/cpu:{difficulty}"
            metrics[f"{prefix}/win_rate"] = statistics.fmean(results)
            metrics[f"{prefix}/window_fights"] = len(results)
            metrics[f"{prefix}/fights"] = _fight_counts[difficulty]
    logging_utils.log(args, metrics, step_key="rollout/step")
    logging.getLogger(__name__).info("SF3 fight metrics: %s", metrics)
    return False


def _image(frame):
    import base64
    import io

    from PIL import Image

    return Image.open(io.BytesIO(base64.b64decode(frame.split(",", 1)[1])))


def megatron_init(args) -> None:
    import torch
    from slime.backends.megatron_utils.data import DataIterator
    from slime.utils.processing_utils import load_processor

    image_processor = load_processor(
        args.hf_checkpoint, trust_remote_code=True
    ).image_processor
    get_next = DataIterator.get_next

    def get_next_with_frames(self, keys):
        batch = get_next(self, keys)
        if batch.get("multimodal_train_inputs"):
            batch["multimodal_train_inputs"] = [
                {
                    key: value.to(torch.cuda.current_device(), non_blocking=True)
                    for key, value in image_processor(
                        images=[_image(entry[FRAME_KEY])], return_tensors="pt"
                    ).items()
                }
                for entry in batch["multimodal_train_inputs"]
            ]
        return batch

    DataIterator.get_next = get_next_with_frames


def sf3_valid_group(args, group) -> bool:
    from slime.utils.types import Sample

    return all(move.status != Sample.Status.ABORTED for seat in group for move in seat)
