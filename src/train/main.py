from modal_dojo import (
    OnlineRollout,
    Qwen3_VL_8B,
    Qwen3_VL_8B_Recipe,
    TrainConfig,
)
from src.train.rollout import (
    sf3_generate,
    sf3_log_rollout,
    sf3_reward_post_process,
    sf3_rollout,
)
from src.train.worker import MAX_WORKERS, WORKER_IMAGE_ENV, build_worker_image
from src.utils import (
    CONTAINER_REGION,
    MAX_TOKENS,
    TEMPERATURE,
    TOP_P,
)

NUM_ROLLOUTS = 50

ROLLOUT_BATCH_SIZE = MAX_WORKERS
N_SAMPLES_PER_PROMPT = 1
GLOBAL_BATCH_SIZE = ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT

model = Qwen3_VL_8B()
recipe = Qwen3_VL_8B_Recipe(
    rollout_function=sf3_rollout,
    custom_generate_function=sf3_generate,
    custom_reward_post_process_function=sf3_reward_post_process,
    custom_rollout_log_function=sf3_log_rollout,
    dynamic_sampling_filter_path="src.train.rollout.sf3_valid_group",
    image_overlay=lambda image: image.add_local_python_source("src", copy=True),
    actor_num_gpus_per_node=1,
    region=CONTAINER_REGION,
    num_rollout=NUM_ROLLOUTS,
    save_interval=5,
    capture_trace=True,
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


if __name__ == "__main__":
    recipe.environment = {
        **recipe.environment,
        WORKER_IMAGE_ENV: build_worker_image(),
    }
    run = config.launch()
    print(f"run id: {run.training_run_id}")
