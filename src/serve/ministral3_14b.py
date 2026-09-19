# https://modal.com/docs/examples/ministral3_inference
# https://docs.vllm.ai/projects/recipes/en/latest/Mistral/Ministral-3-Instruct.html

import base64
import time
import uuid
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
    resolve_move_with_fallback,
)

app = modal.App("sf3-llm")

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.19.0",
        "huggingface-hub==0.36.0",
        "flashinfer-python==0.6.6",
        "qwen-vl-utils==0.0.14",
    )
    .uv_pip_install("transformers==5.5.0")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
        }
    )
)

hf_cache_vol = modal.Volume.from_name("sf3-huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("sf3-vllm-cache", create_if_missing=True)
flashinfer_cache_vol = modal.Volume.from_name(
    "sf3-flashinfer-cache", create_if_missing=True
)

model_name = "mistralai/Ministral-3-14B-Instruct-2512"
model_revision = "1e4bed9a74c1d8af713dd9e2545d69020ced05dc"

max_inputs = max_num_seqs = 64
gpu = "B200:1"
eval_gpu = "H200:1"


@app.cls(
    image=vllm_image,
    gpu=gpu,
    region=CONTAINER_REGION,
    routing_region=ROUTING_REGION,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        "/root/.cache/flashinfer": flashinfer_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=60 * MINUTES,
    timeout=60 * MINUTES,
)
@modal.concurrent(max_inputs=max_inputs, target_inputs=max_inputs // 2)
class Ministral3Server:
    ckpt_path: str = modal.parameter(default="")

    def normalize_messages(self, messages):
        def normalize_image_url(url: str) -> str:
            if url.startswith(("http://", "https://", "data:", "file://")):
                return url
            if url.startswith("/"):
                suffix = Path(url).suffix.lower()
                mime = "image/png" if suffix == ".png" else "image/jpeg"
                return f"data:{mime};base64," + base64.b64encode(
                    Path(url).read_bytes()
                ).decode("utf-8")
            return url

        normalized = []
        for message in messages:
            content = message["content"]
            if not isinstance(content, list):
                normalized.append(message)
                continue

            normalized_content = []
            for item in content:
                if item.get("type") == "image":
                    normalized_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": normalize_image_url(item["image"])},
                        }
                    )
                    continue
                if item.get("type") == "image_url":
                    image_url = item["image_url"]
                    if isinstance(image_url, str):
                        image_url = {"url": image_url}
                    image_url["url"] = normalize_image_url(image_url["url"])
                    normalized_content.append(
                        {
                            "type": "image_url",
                            "image_url": image_url,
                        }
                    )
                    continue
                normalized_content.append(item)

            normalized.append({**message, "content": normalized_content})

        return normalized

    async def _generate(self, messages, sampling_params):
        _, prompts = await self.llm.renderer.render_chat_async(
            [self.normalize_messages(messages)],
            self.chat_params,
        )
        output = None
        async for output in self.llm.generate(
            prompts[0],
            sampling_params,
            request_id=uuid.uuid4().hex,
        ):
            pass
        if output is None:
            raise RuntimeError("vLLM returned no output")
        return output.outputs[0].text

    @modal.enter()
    async def enter(self):
        from vllm import SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.renderers import ChatParams
        from vllm.sampling_params import StructuredOutputsParams
        from vllm.v1.engine.async_llm import AsyncLLM

        load_path = self.ckpt_path or model_name
        revision = model_revision if load_path == model_name else None
        print(f"Loading model from {load_path}")

        self.SamplingParams = SamplingParams
        self.StructuredOutputsParams = StructuredOutputsParams
        self.llm = AsyncLLM.from_engine_args(
            AsyncEngineArgs(
                model=str(load_path),
                revision=revision,
                max_model_len=MAX_CONTEXT_LEN,
                max_num_batched_tokens=8192,
                max_num_seqs=max_num_seqs,
                max_cudagraph_capture_size=max_inputs * 2,
                enable_prefix_caching=True,
                gpu_memory_utilization=0.9,
                disable_log_stats=True,
                limit_mm_per_prompt={"image": 1, "video": 0, "audio": 0},
                kv_cache_dtype="fp8",
                async_scheduling=True,
                tokenizer_mode="mistral",
                config_format="mistral",
                load_format="mistral",
            )
        )

        self.chat_params = ChatParams(
            chat_template_kwargs={
                "add_generation_prompt": True,
                "continue_final_message": False,
                "tokenize": True,
            }
        )
        self.sampling_params_kwargs = {
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
        }

        messages, _, _, _, _, available_moves = create_random_messages()

        _ = await self._generate(
            messages,
            self.SamplingParams(
                **self.sampling_params_kwargs,
                structured_outputs=self.StructuredOutputsParams(
                    choice=available_moves,
                ),
            ),
        )
        await self.llm.reset_mm_cache()

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

        sampling_params = self.SamplingParams(
            **self.sampling_params_kwargs,
            structured_outputs=self.StructuredOutputsParams(
                choice=available_moves,
            ),
        )

        output = await self._generate(
            messages,
            sampling_params,
        )
        move_name = output.strip()

        move_sequence, resolved_move_name = resolve_move_with_fallback(
            character, move_name, side
        )
        if resolved_move_name == "No-Move":
            print(f"Invalid move: {move_name}")
        return move_sequence, resolved_move_name

    @modal.exit()
    def exit(self):
        self.llm.shutdown()


@app.function(
    routing_region=ROUTING_REGION,
    timeout=15 * MINUTES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
async def test_ministral(n_samples: int):
    llm = Ministral3Server()
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
    await test_ministral.remote.aio(n_samples)
