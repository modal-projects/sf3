# https://modal.com/docs/examples/sglang_low_latency
# https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html
# https://huggingface.co/google/gemma-4-31B-it#1-sampling-parameters

import json
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
    resolve_move_with_fallback,
)

app = modal.App("sf3-llm")

sglang_image = (
    modal.Image.from_registry("modalresearch/sglang:nightly-dev-cu13-20260619-patched")
    .entrypoint([])
    .run_commands("rm -rf /root/.cache/huggingface")
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

model_name = "google/gemma-4-31B-it"
model_revision = "3548789868c5356dbf307c98e6f609007b82b3eb"
draft_model_name = "z-lab/gemma-4-31B-it-DFlash"
draft_model_revision = "eabd648301ce28583cc14757912e5e0f84e152e1"

max_inputs = max_num_seqs = 16
gpu = "B200:1"
eval_gpu = gpu


def _unique_move_from_json_prefix(raw: str, available_moves: list[str]) -> str | None:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        try:
            value = json.loads(raw + '"')
        except json.JSONDecodeError:
            return None
        if not isinstance(value, str):
            return None
        matches = [move for move in available_moves if move.startswith(value)]
        return matches[0] if len(matches) == 1 else None
    return value if isinstance(value, str) and value in available_moves else None


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
class Gemma4Server:
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

    @staticmethod
    def _move_schema(available_moves: list[str]) -> str:
        return json.dumps({"type": "string", "enum": available_moves})

    @modal.enter()
    async def enter(self):
        import sglang as sgl

        load_path = self.ckpt_path or model_name
        revision = model_revision if load_path == model_name else None
        print(f"Loading model from {load_path}")

        self.llm = sgl.Engine(
            model_path=str(load_path),
            revision=revision,
            context_length=MAX_CONTEXT_LEN,
            attention_backend="triton",
            chunked_prefill_size=8192,
            max_prefill_tokens=8192,
            max_running_requests=max_num_seqs,
            cuda_graph_max_bs_decode=max_inputs * 2,
            cuda_graph_max_bs_prefill=max_inputs * 2,
            disable_cuda_graph_padding=True,
            mem_fraction_static=0.85,
            grammar_backend="xgrammar",
            enable_multimodal=True,
            trust_remote_code=True,
            speculative_algorithm="DFLASH",
            speculative_draft_model_path=draft_model_name,
            speculative_draft_model_revision=draft_model_revision,
            speculative_dflash_block_size=16,
            speculative_draft_attention_backend="fa4",
        )
        self.tokenizer = self.llm.tokenizer_manager.tokenizer

        self.sampling_params = {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "max_new_tokens": MAX_TOKENS,
        }

        messages, _, _, _, _, available_moves = create_random_messages()
        prompt, images = self.prepare_request(messages)
        warmup_params = {
            **self.sampling_params,
            "json_schema": self._move_schema(available_moves),
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
            "json_schema": self._move_schema(available_moves),
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
                move_name = _unique_move_from_json_prefix(
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
            raw = last_output["text"].strip()
            try:
                move_name = json.loads(raw) if raw.startswith('"') else raw
            except json.JSONDecodeError:
                move_name = raw.strip('"')
            if not isinstance(move_name, str):
                move_name = str(move_name)

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
async def test_gemma(n_samples: int):
    llm = Gemma4Server()
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
    await test_gemma.remote.aio(n_samples)
