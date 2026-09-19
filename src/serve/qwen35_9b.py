# https://modal.com/docs/examples/sglang_low_latency
# https://huggingface.co/Qwen/Qwen3.5-9B
# https://cookbook.sglang.io/autoregressive/Qwen/Qwen3.6

import time
import uuid

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

app = modal.App("sf3-llm")

sglang_image = (
    modal.Image.from_registry("modalresearch/sglang:nightly-dev-cu13-20260619-patched")
    .entrypoint([])
    .run_commands("rm -rf /root/.cache/huggingface /root/.cache/flashinfer")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "CUDA_ENABLE_COREDUMP_ON_EXCEPTION": "0",
        }
    )
)

hf_cache_vol = modal.Volume.from_name("sf3-huggingface-cache", create_if_missing=True)
flashinfer_cache_vol = modal.Volume.from_name(
    "sf3-flashinfer-cache", create_if_missing=True
)

model_name = "Qwen/Qwen3.5-9B"
model_revision = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"

max_inputs = max_num_seqs = 32
gpu = "B200:1"
eval_gpu = "H200:1"


def _unique_move_from_prefix(raw: str, available_moves: list[str]) -> str | None:
    if not raw:
        return None
    matches = [move for move in available_moves if move.startswith(raw)]
    return matches[0] if len(matches) == 1 else None


@app.cls(
    image=sglang_image,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/flashinfer": flashinfer_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu=gpu,
    region=CONTAINER_REGION,
    routing_region=ROUTING_REGION,
    scaledown_window=60 * MINUTES,
    timeout=60 * MINUTES,
)
@modal.concurrent(max_inputs=max_inputs, target_inputs=max_inputs // 2)
class Qwen35Server:
    ckpt_path: str = modal.parameter(default="")

    def prepare_request(self, messages: list[dict]) -> tuple[str, list[str]]:
        images = [
            item["image"]
            for message in messages
            if isinstance(message["content"], list)
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
        import torch

        load_path = self.ckpt_path or model_name
        revision = model_revision if load_path == model_name else None
        blackwell = torch.cuda.get_device_capability()[0] >= 10
        print(f"Loading model from {load_path} (blackwell={blackwell})")

        self.llm = sgl.Engine(
            model_path=str(load_path),
            revision=revision,
            context_length=MAX_CONTEXT_LEN,
            chunked_prefill_size=8192,
            max_running_requests=max_num_seqs,
            cuda_graph_max_bs_decode=max_inputs * 2,
            mem_fraction_static=0.7,
            grammar_backend="xgrammar",
            attention_backend="trtllm_mha" if blackwell else "fa3",
            linear_attn_prefill_backend="flashinfer",
            linear_attn_decode_backend="flashinfer",
            mamba_ssm_dtype="bfloat16",
            mm_attention_backend="fa4" if blackwell else "fa3",
            trust_remote_code=True,
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

        _ = await self.llm.async_generate(
            prompt,
            image_data=images,
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

        sampling_params = {
            **self.sampling_params,
            "regex": move_regex(available_moves),
        }
        prompt, images = self.prepare_request(messages)
        request_id = uuid.uuid4().hex
        generator = await self.llm.async_generate(
            prompt,
            image_data=images,
            sampling_params=sampling_params,
            stream=True,
            rid=request_id,
        )
        last_output = None
        move_name = None
        try:
            async for output in generator:
                last_output = output
                if output["meta_info"].get("finish_reason") is not None:
                    break
                move_name = _unique_move_from_prefix(
                    output["text"].strip(), available_moves
                )
                if move_name is not None:
                    break
        finally:
            try:
                if (
                    last_output is None
                    or last_output["meta_info"].get("finish_reason") is None
                ):
                    self.llm.tokenizer_manager.abort_request(request_id)
            finally:
                await generator.aclose()

        if last_output is None:
            raise RuntimeError("SGLang stream produced no output")

        if move_name is None:
            move_name = last_output["text"].strip()

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
    timeout=15 * MINUTES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
async def test_qwen35(n_samples: int):
    llm = Qwen35Server()
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
async def main(n_samples: int = 100):
    await test_qwen35.remote.aio(n_samples)
