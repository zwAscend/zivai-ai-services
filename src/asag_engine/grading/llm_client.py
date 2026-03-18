import os
import time
from functools import lru_cache
from typing import Optional

class LLMClient:
    def generate(
        self,
        system_text: str,
        user_text: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        raise NotImplementedError

class MindNLPLocalClient(LLMClient):
    def __init__(self, model_id: str, max_new_tokens: int = 768, ms_mode: Optional[str] = None):
        import mindspore as ms
        from mindnlp.transformers import AutoTokenizer, AutoModelForCausalLM

        self.ms = ms
        self.model_id = model_id
        ms_mode = ms_mode or os.getenv("MS_MODE", "GRAPH_MODE")
        if ms_mode.upper() == "PYNATIVE_MODE":
            self.ms.set_context(mode=self.ms.PYNATIVE_MODE)
        else:
            self.ms.set_context(mode=self.ms.GRAPH_MODE)

        self._configure_device_target()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.max_new_tokens = int(max_new_tokens)
        self.max_input_tokens = int(os.getenv("MAX_INPUT_TOKENS", "0"))

    def _configure_device_target(self) -> None:
        target = os.getenv("MS_DEVICE_TARGET", "GPU").strip().upper()
        strict = os.getenv("MS_STRICT_DEVICE", "true").strip().lower() == "true"

        try:
            # Newer MindSpore prefers set_device.
            if hasattr(self.ms, "set_device"):
                self.ms.set_device(target)
            else:
                self.ms.set_context(device_target=target)
        except Exception as e:
            if strict:
                raise RuntimeError(
                    f"MindSpore failed to initialize device target '{target}'. "
                    "Set MS_DEVICE_TARGET=CPU to run without GPU, or fix MindSpore GPU runtime."
                ) from e

            # Non-strict fallback: use CPU.
            if hasattr(self.ms, "set_device"):
                self.ms.set_device("CPU")
            else:
                self.ms.set_context(device_target="CPU")
            print(
                f"[WARN] MindSpore device '{target}' unavailable. Falling back to CPU. "
                "Set MS_STRICT_DEVICE=true to fail instead."
            )

    def _prompt(self, system_text: str, user_text: str) -> str:
        return f"<|system|>\n{system_text}\n<|user|>\n{user_text}\n<|assistant|>\n"

    def _render_prompt(self, system_text: str, user_text: str) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ]
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as exc:
                print(f"[llm] chat template unavailable, falling back to raw prompt: {exc}")
        return self._prompt(system_text, user_text)

    def generate(
        self,
        system_text: str,
        user_text: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        prompt = self._render_prompt(system_text, user_text)
        tokenizer_kwargs = {"return_tensors": "ms"}
        if self.max_input_tokens > 0:
            tokenizer_kwargs.update({"truncation": True, "max_length": self.max_input_tokens})

        generation_tokens = int(max_new_tokens or self.max_new_tokens)
        started = time.perf_counter()
        inputs = self.tokenizer(prompt, **tokenizer_kwargs)
        input_shape = getattr(inputs["input_ids"], "shape", ())
        input_tokens = input_shape[-1] if input_shape else "?"
        print(
            "[llm] generation start "
            f"model={self.model_id} input_tokens={input_tokens} max_new_tokens={generation_tokens}"
        )
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=generation_tokens,
            do_sample=False
        )
        generated_ids = output_ids[0]
        input_ids = inputs.get("input_ids")
        if input_ids is not None:
            prompt_tokens = input_ids.shape[-1]
            output_shape = getattr(generated_ids, "shape", ())
            if output_shape and output_shape[-1] > prompt_tokens:
                generated_ids = generated_ids[prompt_tokens:]

        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        if "<|assistant|>" in text:
            text = text.split("<|assistant|>", 1)[-1].strip()
        text = text.strip()
        elapsed = time.perf_counter() - started
        print(f"[llm] generation done seconds={elapsed:.2f} output_chars={len(text)}")
        return text

class PanguNLPClient(MindNLPLocalClient):
    # If you have a Pangu checkpoint compatible with MindNLP transformers,
    # set MODEL_ID accordingly and set LLM_PROVIDER=pangu.
    pass

@lru_cache(maxsize=1)
def build_llm_client() -> LLMClient:
    provider = os.getenv("LLM_PROVIDER", "mindnlp").strip().lower()
    model_id = os.getenv("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
    max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "768"))

    if provider == "pangu":
        return PanguNLPClient(model_id=model_id, max_new_tokens=max_new_tokens)
    return MindNLPLocalClient(model_id=model_id, max_new_tokens=max_new_tokens)
