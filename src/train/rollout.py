import asyncio
import random
import traceback
from collections import deque
from itertools import zip_longest

from modal_training_gym import Qwen3_VL_8B

from src.env import EnvironmentConfig, create_environment
from src.utils import (
    CHARACTER_MAPPING,
    HEALTH_MAX,
    RECENT_MOVE_LIMIT,
    FrameEncoder,
    create_messages,
    move_regex,
    player_state,
    resolve_move_with_fallback,
)

MODEL = Qwen3_VL_8B()
ROSTER = tuple(CHARACTER_MAPPING.values())
OUTFIT = SUPER_ART = 1
FRAME_KEY = "sf3_frame"
IMAGE_PAD = "<|image_pad|>"
GAMMA = 0.9

_fights: dict[tuple[int, int], asyncio.Task] = {}
_image_tokens_by_frame_size: dict[tuple[int, int], int] = {}


async def _move(
    args, sample, sampling_params, seat, fighters, frame, frame_size, recent
):
    from slime.rollout.sglang_rollout import GenerateState
    from slime.utils.http_utils import post
    from slime.utils.types import Sample

    state = GenerateState(args)
    if state.aborted:
        raise RuntimeError("rollout aborted")
    messages, available = create_messages(
        fighters[1 - seat], fighters[seat], [frame], recent[seat]
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
    index = sample.index - sample.index % 2 + seat
    move = Sample(
        group_index=sample.group_index,
        index=index,
        rollout_id=index,
        prompt=prompt_text,
        tokens=prompt_ids,
        multimodal_train_inputs={FRAME_KEY: frame},
    )
    output = await post(
        f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate",
        {
            "text": prompt_text,
            "image_data": [frame],
            "sampling_params": {**sampling_params, "regex": move_regex(available)},
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
    return move


async def _play_fight(args, sample, sampling_params):
    characters = random.sample(ROSTER, 2)
    identities = [{"character": c, "superArt": SUPER_ART} for c in characters]
    env = await asyncio.to_thread(
        create_environment,
        EnvironmentConfig(
            characters=tuple(characters),
            outfits=(OUTFIT, OUTFIT),
            super_arts=(SUPER_ART, SUPER_ART),
            step_ratio=6,
        ),
    )
    try:
        observation, _ = await asyncio.to_thread(env.reset)
        encoder = FrameEncoder()
        recent = [deque(maxlen=RECENT_MOVE_LIMIT), deque(maxlen=RECENT_MOVE_LIMIT)]
        moves = [[], []]
        damage, rounds, round_index = [], [], 0
        while True:
            fighters = [
                player_state(observation, identities[seat], f"P{seat + 1}")
                for seat in range(2)
            ]
            pixels = observation["frame"]
            frame_size = pixels.shape
            frame = encoder.data_url(pixels)
            turn = await asyncio.gather(
                *(
                    _move(
                        args,
                        sample,
                        sampling_params,
                        seat,
                        fighters,
                        frame,
                        frame_size,
                        recent,
                    )
                    for seat in range(2)
                )
            )
            buttons = []
            for seat, move in enumerate(turn):
                moves[seat].append(move)
                move_buttons, move_name = resolve_move_with_fallback(
                    characters[seat],
                    MODEL.parse_response(move.response).content,
                    fighters[seat].side,
                )
                recent[seat].append(move_name)
                buttons.append(move_buttons)
            turn_damage = 0.0
            for p1_button, p2_button in zip_longest(*buttons, fillvalue=0):
                observation, step_damage, terminated, _, info = await asyncio.to_thread(
                    env.step, {"agent_0": p1_button, "agent_1": p2_button}
                )
                turn_damage += step_damage
                if terminated or info["round_done"]:
                    break
            damage.append(turn_damage)
            rounds.append(round_index)
            if info["round_done"]:
                round_index += 1
                for seat_recent in recent:
                    seat_recent.clear()
            if terminated:
                break
    finally:
        await asyncio.to_thread(env.close)
    returns, G = [0.0] * len(damage), 0.0
    for t in reversed(range(len(damage))):
        if t + 1 < len(damage) and rounds[t + 1] != rounds[t]:
            G = 0.0
        G = damage[t] + GAMMA * G
        returns[t] = G / HEALTH_MAX
    for seat_moves, sign in zip(moves, (1, -1)):
        for move, G in zip(seat_moves, returns):
            move.reward = sign * G
    return moves


async def sf3_generate(args, sample, sampling_params):
    from slime.utils.types import Sample

    key = (sample.group_index, sample.index // 2)
    fight = _fights.pop(key, None)
    if fight is None:
        fight = _fights[key] = asyncio.create_task(
            _play_fight(args, sample, sampling_params)
        )
    try:
        return (await asyncio.shield(fight))[sample.index % 2]
    except Exception:
        traceback.print_exc()
        sample.status = Sample.Status.ABORTED
        return [sample]


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
