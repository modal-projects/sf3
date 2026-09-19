import time
from pathlib import Path

import modal

from src.utils import (
    CONTAINER_REGION,
    MAX_CONTEXT_LEN,
    MAX_TOKENS,
    MINUTES,
    ROUTING_REGION,
    create_random_messages,
    get_available_instructions_for_character,
    move_regex,
    resolve_move_with_fallback,
)

APP_NAME = "sf3-qwen3-vl-8b"

app = modal.App(APP_NAME)

model_name = "Qwen/Qwen3-VL-8B-Instruct"
model_revision = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"

sglang_image = (
    modal.Image
    .from_registry("lmsysorg/sglang:v0.5.12.post1-cu130")
    .entrypoint([])
    .run_commands("rm -rf /root/.cache/huggingface /root/.cache/flashinfer")
    .uv_pip_install("qwen-vl-utils==0.0.14")
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
    })
)

hf_cache_vol = modal.Volume.from_name("sf3-huggingface-cache", create_if_missing=True)
flashinfer_cache_vol = modal.Volume.from_name(
    "sf3-flashinfer-cache", create_if_missing=True
)

CHECKPOINTS_MOUNT = "/checkpoints"
# training-gym's slime launcher names this volume
# f"slime-{recipe_class_name.lower()}-checkpoints".
CHECKPOINTS_VOLUME = "slime-qwen3_vl_8b_recipe-checkpoints"

cache_volumes = {
    "/root/.cache/huggingface": hf_cache_vol,
    "/root/.cache/flashinfer": flashinfer_cache_vol,
    CHECKPOINTS_MOUNT: modal.Volume.from_name(
        CHECKPOINTS_VOLUME, create_if_missing=True
    ).with_mount_options(read_only=True),
}


def latest_hf_export_on_volume() -> str | None:
    exports = list(Path(CHECKPOINTS_MOUNT).glob("*/*_hf"))
    return str(max(exports, key=lambda path: path.stat().st_mtime)) if exports else None


max_inputs = max_num_seqs = 32
gpu = "B200:1"
eval_gpu = "H100:1"


@app.cls(
    image=sglang_image,
    gpu=gpu,
    region=CONTAINER_REGION,
    routing_region=ROUTING_REGION,
    volumes=cache_volumes,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=60 * MINUTES,
    timeout=60 * MINUTES,
)
@modal.concurrent(max_inputs=max_inputs, target_inputs=max_inputs // 2)
class Qwen3VLServer:
    ckpt_path: str = modal.parameter(default="")

    def prepare_request(self, messages: list[dict]) -> tuple[str, list[str]]:
        images = [
            item["image"]
            for message in messages
            if isinstance(message.get("content"), list)
            for item in message["content"]
            if item.get("type") == "image"
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            return_dict=False,
            enable_thinking=False,
        )
        return prompt, images

    @modal.enter()
    async def enter(self):
        import sglang as sgl

        load_path = self.ckpt_path or latest_hf_export_on_volume() or model_name
        revision = model_revision if load_path == model_name else None
        print(f"Loading model from {load_path}")
        self.llm = sgl.Engine(
            model_path=str(load_path),
            revision=revision,
            context_length=MAX_CONTEXT_LEN,
            chunked_prefill_size=8192,
            max_running_requests=max_num_seqs,
            cuda_graph_max_bs=max_inputs * 2,
            mem_fraction_static=0.9,
            grammar_backend="xgrammar",
        )
        self.tokenizer = self.llm.tokenizer_manager.tokenizer
        self.sampling_params = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
            "max_new_tokens": MAX_TOKENS,
        }

        messages, _, _, _, _, available_moves = create_random_messages()
        prompt, images = self.prepare_request(messages)
        warmup_params = {
            **self.sampling_params,
            "regex": move_regex(available_moves),
        }
        await self.llm.async_generate(
            prompt,
            image_data=images or None,
            sampling_params=warmup_params,
        )

    @modal.method()
    async def boot(self):
        pass

    @modal.method()
    async def chat(
        self,
        messages: list[dict],
        character: str,
        super_art: int,
        super_count: int,
        side: int,
        available_moves: list[str] | None = None,
    ) -> tuple[list[int], str]:
        if available_moves is None:
            available_moves = get_available_instructions_for_character(
                character, super_art, super_count
            )
        prompt, images = self.prepare_request(messages)
        output = await self.llm.async_generate(
            prompt,
            image_data=images or None,
            sampling_params={
                **self.sampling_params,
                "regex": move_regex(available_moves),
            },
        )
        move_name = output["text"].strip()

        move_sequence, resolved_move_name = resolve_move_with_fallback(
            character, move_name, side
        )
        if resolved_move_name == "No-Move":
            print(f"Invalid move: {move_name}")
        return move_sequence, resolved_move_name

    @modal.exit()
    async def exit(self):
        self.llm.shutdown()


@app.function(
    routing_region=ROUTING_REGION,
    timeout=60 * MINUTES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
async def test_qwen3(n_samples: int, ckpt_path: str = ""):
    llm = Qwen3VLServer(ckpt_path=ckpt_path)
    await llm.boot.remote.aio()

    ms_per_move = []
    for sample_idx in range(n_samples):
        messages, character, super_art, super_count, side, available_moves = (
            create_random_messages()
        )
        start_time = time.perf_counter()
        buttons, move = await llm.chat.remote.aio(
            messages, character, super_art, super_count, side, available_moves
        )
        elapsed = (time.perf_counter() - start_time) * 1000
        ms_per_move.append(elapsed)
        print(f"Sample {sample_idx}: {elapsed:.2f}ms, {move}, {buttons}")

    percentiles = [50, 90, 95, 99]
    sorted_ms = sorted(ms_per_move)
    results = {}
    for p in percentiles:
        idx = int(len(sorted_ms) * p / 100)
        idx = min(max(idx - 1, 0), len(sorted_ms) - 1)
        results[p] = sorted_ms[idx]
    print("--------------------------------")
    print("Latency per move percentiles (ms):")
    for p in percentiles:
        print(f"  p{p}: {results[p]:.2f}ms")
    print("--------------------------------")


@app.local_entrypoint()
async def main(n_samples: int = 100, ckpt_path: str = ""):
    await test_qwen3.remote.aio(n_samples, ckpt_path=ckpt_path)
