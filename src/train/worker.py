import asyncio
import gzip
import json
import logging
import os
from collections import deque
from dataclasses import asdict
from pathlib import Path

import httpx
import modal

from src.env import EnvironmentConfig, create_environment
from src.utils import (
    CONTAINER_REGION,
    GB,
    MINUTES,
    RECENT_MOVE_LIMIT,
    FrameEncoder,
    create_gameplay_image,
    model_reaction_frames,
    player_state,
    resolve_move_with_fallback,
)

APP_NAME = "sf3-training-sandboxes"
MAX_WORKERS = 32
FRAME_KEY = "sf3_frame"
WORKER_IMAGE_ENV = "SF3_WORKER_IMAGE_ID"
WORKER_INPUT_ENV = "SF3_WORKER_INPUT"
RESULT_PATH = "/tmp/sf3-rollout.json.gz"
WORKER_TIMEOUT = 60 * MINUTES

image = create_gameplay_image(
    extra_python_packages=("httpx==0.28.1", "modal==1.6.1"),
    copy=True,
    add_python_source=True,
)
_slots = asyncio.Semaphore(MAX_WORKERS)


def build_worker_image() -> str:
    image.build(modal.App.lookup(APP_NAME, create_if_missing=True))
    return image.object_id


async def run_arcade_sandbox(
    config: EnvironmentConfig,
    endpoint: str,
    sample: dict,
    sampling_params: dict,
    inference_latency_ms: float,
) -> dict:
    payload = {
        "config": asdict(config),
        "endpoint": endpoint,
        "sample": sample,
        "sampling_params": sampling_params,
        "inference_latency_ms": inference_latency_ms,
    }
    async with _slots:
        app = await modal.App.lookup.aio(APP_NAME)
        sandbox = await modal.Sandbox.create.aio(
            "sleep",
            str(WORKER_TIMEOUT),
            app=app,
            image=modal.Image.from_id(os.environ[WORKER_IMAGE_ENV]),
            cpu=1,
            memory=2 * GB,
            region=CONTAINER_REGION,
            runtime="gvisor",
            timeout=WORKER_TIMEOUT,
            workdir="/root",
            env={WORKER_INPUT_ENV: json.dumps(payload)},
        )
        try:
            process = await sandbox.exec.aio(
                "python", "-m", "src.train.worker", timeout=WORKER_TIMEOUT
            )
            if await process.wait.aio() != 0:
                raise RuntimeError(
                    f"Arcade sandbox {sandbox.object_id} failed: "
                    f"{await process.stderr.read.aio()}"
                )
            data = await sandbox.filesystem.read_bytes.aio(RESULT_PATH)
            result = json.loads(gzip.decompress(data))
            result["sandbox_id"] = sandbox.object_id
            return result
        finally:
            await asyncio.shield(sandbox.terminate.aio())


def play_arcade(
    config: EnvironmentConfig,
    endpoint: str,
    sample: dict,
    sampling_params: dict,
    inference_latency_ms: float,
) -> dict:
    env = create_environment(config)
    try:
        observation, _ = env.reset()
        encoder = FrameEncoder()
        recent = deque(maxlen=RECENT_MOVE_LIMIT)
        rounds, results = [], []
        moves, invalid, damage, durations = [], [], [], []
        with httpx.Client(timeout=60) as client:
            while True:
                identities = env.read_match_identity()
                fighters = [
                    player_state(
                        observation, identities[f"player{seat + 1}"], f"P{seat + 1}"
                    )
                    for seat in range(2)
                ]
                pixels = observation["frame"]
                frame = encoder.data_url(pixels)
                payload = {
                    "sample": sample,
                    "sampling_params": sampling_params,
                    "fighters": [asdict(fighter) for fighter in fighters],
                    "frame": frame,
                    "frame_size": pixels.shape,
                    "recent": list(recent),
                }
                for attempt in range(3):
                    try:
                        response = client.post(f"{endpoint}/move", json=payload)
                        response.raise_for_status()
                        break
                    except httpx.TransportError:
                        if attempt == 2:
                            raise
                        logging.getLogger(__name__).warning(
                            "Retrying inference transport for the same observation"
                        )
                generated = response.json()
                name = generated["name"]
                bad = name not in generated["available"]
                buttons, move_name = resolve_move_with_fallback(
                    fighters[0].character,
                    name if not bad else "No-Move",
                    fighters[0].side,
                )
                delay = model_reaction_frames(buttons, inference_latency_ms)
                observation, hp_diff, terminated, truncated, info = env.step_frames(
                    [{"agent_0": button} for button in [0] * delay + buttons]
                )
                if truncated:
                    raise RuntimeError("Arcade run truncated before normal completion")
                move = generated["sample"]
                move["multimodal_train_inputs"] = {FRAME_KEY: frame}
                moves.append(move)
                invalid.append(bad)
                damage.append(hp_diff)
                durations.append(info["frames"])
                recent.append(move_name)
                if info["round_done"]:
                    rounds.append(
                        {
                            "moves": moves,
                            "invalid": invalid,
                            "damage": damage,
                            "durations": durations,
                            "winner": info["round_winner"],
                        }
                    )
                    moves, invalid, damage, durations = [], [], [], []
                    recent.clear()
                if info["stage_done"] or terminated:
                    results.append(info["stage_done"] or info["winner"] == "P1")
                if terminated:
                    return {"rounds": rounds, "results": results}
    finally:
        env.close()


if __name__ == "__main__":
    request = json.loads(os.environ[WORKER_INPUT_ENV])
    request["config"] = EnvironmentConfig(**request["config"])
    result = play_arcade(**request)
    Path(RESULT_PATH).write_bytes(
        gzip.compress(json.dumps(result).encode("utf-8"), compresslevel=1)
    )
