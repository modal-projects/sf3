import time

from modal_training_gym import (
    OnlineRollout,
    Qwen3_VL_8B,
    Qwen3_VL_8B_Recipe,
    TrainConfig,
    convert_megatron_checkpoint_to_hf,
)

from src.train.rollout import sf3_generate
from src.utils import MAX_TOKENS, create_gameplay_image

NUM_ROLLOUTS = 10

ROLLOUT_BATCH_SIZE = 4
N_SAMPLES_PER_PROMPT = 2

model = Qwen3_VL_8B()
recipe = Qwen3_VL_8B_Recipe(
    custom_generate_function=sf3_generate,
    dynamic_sampling_filter_path="src.train.rollout.sf3_valid_group",
    image_overlay=lambda image: create_gameplay_image(
        base_image=image,
        copy=True,
        add_python_source=True,
    ),
    num_rollout=NUM_ROLLOUTS,
    rollout_batch_size=ROLLOUT_BATCH_SIZE,
    n_samples_per_prompt=N_SAMPLES_PER_PROMPT,
    global_batch_size=ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT,
    rollout_max_response_len=MAX_TOKENS,
    extra_config={
        **Qwen3_VL_8B_Recipe().extra_config,
        "micro_batch_size": 8,  # a step is ~14k moves; mbs=1 took ~60 min to train
        "rewards_normalization": False,
        "custom_megatron_init_path": "src.train.rollout.megatron_init",
    },
)

config = TrainConfig(
    model=model,
    dataset=OnlineRollout(n_rows=ROLLOUT_BATCH_SIZE),
    recipe=recipe,
)

with config.launch() as run:
    print(f"run id: {run.training_run_id}")
    checkpoint = None
    while True:
        done = run.done()
        latest = run.latest_checkpoint()
        if latest is not None and latest != checkpoint:
            checkpoint = latest
            print(f"new checkpoint: {checkpoint.path}")
        if done:
            break
        time.sleep(30)
    if checkpoint is None:
        raise RuntimeError("run produced no checkpoint")
    print(convert_megatron_checkpoint_to_hf(checkpoint, model).path)
