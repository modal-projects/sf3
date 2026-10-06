import time
from concurrent.futures import ThreadPoolExecutor

import modal
from modal_dojo import (
    OnlineRollout,
    Qwen3_VL_8B,
    Qwen3_VL_8B_Recipe,
    TrainConfig,
    convert_megatron_checkpoint_to_hf,
)

from src.eval.main import app as eval_app
from src.eval.main import cache_volume, orchestrate
from src.serve import POLICY_MODEL_KEY
from src.serve.qwen3_vl_8b import CHECKPOINTS_MOUNT
from src.train.rollout import sf3_generate
from src.utils import MAX_TOKENS, TEMPERATURE, TOP_P, create_gameplay_image

NUM_ROLLOUTS = 100

ROLLOUT_BATCH_SIZE = 32
N_SAMPLES_PER_PROMPT = 2
GLOBAL_BATCH_SIZE = ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT

model = Qwen3_VL_8B()
recipe = Qwen3_VL_8B_Recipe(
    custom_generate_function=sf3_generate,
    dynamic_sampling_filter_path="src.train.rollout.sf3_valid_group",
    image_overlay=lambda image: create_gameplay_image(
        base_image=image,
        copy=True,
        add_python_source=True,
    ),
    actor_num_gpus_per_node=8,
    num_rollout=NUM_ROLLOUTS,
    save_interval=10,
    rollout_batch_size=ROLLOUT_BATCH_SIZE,
    n_samples_per_prompt=N_SAMPLES_PER_PROMPT,
    global_batch_size=GLOBAL_BATCH_SIZE,
    rollout_max_response_len=MAX_TOKENS,
    rollout_temperature=TEMPERATURE,
    rollout_top_p=TOP_P,
    extra_config={
        **Qwen3_VL_8B_Recipe().extra_config,
        "micro_batch_size": 8,
        "custom_megatron_init_path": "src.train.rollout.megatron_init",
    },
)

config = TrainConfig(
    model=model,
    dataset=OnlineRollout(n_rows=ROLLOUT_BATCH_SIZE),
    recipe=recipe,
)


cache_volume_url = (
    f"https://modal.com/storage/{modal.Workspace.from_context().hydrate().name}"
    f"/{modal.Environment.from_context().hydrate().name}/{cache_volume.name}"
)


def eval_checkpoint(checkpoint):
    hf_checkpoint = convert_megatron_checkpoint_to_hf(checkpoint, model)
    print(f"converted checkpoint: {hf_checkpoint.path}")
    player = f"{CHECKPOINTS_MOUNT}/{hf_checkpoint.path_relative_to_volume}"
    for opponent in ("cpu:1", POLICY_MODEL_KEY):
        report = orchestrate.remote(players=(player, opponent))
        print(
            f"{checkpoint.name} vs {opponent}: "
            f"{cache_volume_url}/{report['config']['output_dir']}"
        )


with config.launch() as run, eval_app.run(), ThreadPoolExecutor() as evals:
    print(f"run id: {run.training_run_id}")
    checkpoint, pending = None, []
    while True:
        done = run.done()
        latest = run.latest_checkpoint()
        if latest is not None and latest != checkpoint:
            checkpoint = latest
            print(f"new checkpoint: {checkpoint.path}")
            pending.append((checkpoint.name, evals.submit(eval_checkpoint, checkpoint)))
        if done:
            break
        time.sleep(30)
    if checkpoint is None:
        raise RuntimeError("run produced no checkpoint")
    for name, future in pending:
        try:
            future.result()
        except Exception as exc:
            print(f"{name} eval failed: {type(exc).__name__}: {exc}")
