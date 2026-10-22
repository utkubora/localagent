from typing import Any

from ..tools.baseTool import Tool, ToolCall, ToolResult, Turn, _new_id
from .common import SYSTEM_PROMPT, Backend, coerce_args, scrape_turn


def _best_device_map() -> str:
    """Prefer a real accelerator over transformers' generic placement heuristic."""
    import torch

    if torch.cuda.is_available():
        print("Using CUDA backend for local model inference.")
        return "cuda"
    if torch.backends.mps.is_available():
        print("Using MPS backend for local model inference.")
        return "mps"
    print("Using CPU backend for local model inference. This will be slow.")
    return "auto"


class TransformersBackend(Backend):
    """A local Hugging Face causal LM.

    Needs a checkpoint whose chat template supports tools — Qwen2.5/Qwen3,
    Llama 3.1+, Mistral/Ministral, Hermes, SmolLM3, Command-R and friends. A
    base model, or a chat model with no tool-use template, will raise on the
    first apply_chat_template call.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        system: str = SYSTEM_PROMPT,
        device_map: str | None = None,
        dtype: str = "auto",
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
        do_sample: bool = True,
        stream: bool = False,
        quantization_config: Any = None,
        model_kwargs: dict | None = None,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device_map is None:
            device_map = _best_device_map()

        kwargs: dict[str, Any] = {"device_map": device_map, **(model_kwargs or {})}
        if quantization_config is not None:  # e.g. BitsAndBytesConfig(load_in_4bit=True)
            kwargs["quantization_config"] = quantization_config

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype, **kwargs)
        except TypeError:  # transformers < 4.56 spells it torch_dtype
            self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=dtype, **kwargs)
        self.model.eval()

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.stream = stream
        self.messages: list[dict] = []
        if system:
            self.messages.append({"role": "system", "content": system})

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def generate(self, tools: list[Tool]) -> Turn:
        import torch

        # The template renders the tool schemas into the prompt itself, so the
        # whole conversation is re-encoded from scratch on every step.
        inputs = self.tokenizer.apply_chat_template(
            self.messages,
            tools=[t.openai_spec for t in tools],
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        streamer = None
        if self.stream:
            from transformers import TextStreamer

            streamer = TextStreamer(self.tokenizer, skip_prompt=True)

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature if self.do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                streamer=streamer,
            )

        raw = self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=False)
        turn = self.parse(raw, inputs["input_ids"][0])
        self.messages.append(self._assistant_message(turn))
        return turn

    def parse(self, raw: str, prefix: Any = None) -> Turn:
        """Prefer the tokenizer's own response template; fall back to regex.

        transformers ships `parse_response` for checkpoints that define a
        response_template. Where that's missing we scrape the text ourselves.
        """
        try:
            parsed = self.tokenizer.parse_response(raw, prefix=prefix)
            calls = []
            for entry in parsed.get("tool_calls") or []:
                fn = entry.get("function", entry) if isinstance(entry, dict) else {}
                if fn.get("name"):
                    calls.append(ToolCall(_new_id(), fn["name"], coerce_args(fn.get("arguments"))))
            return Turn(
                text=(parsed.get("content") or "").strip(),
                thinking=(parsed.get("thinking") or "").strip(),
                tool_calls=calls,
            )
        except Exception:
            pass  # no response_template on this tokenizer, or parsing failed

        text = self.tokenizer.decode(
            self.tokenizer.encode(raw, add_special_tokens=False), skip_special_tokens=True
        )
        return scrape_turn(text)

    @staticmethod
    def _assistant_message(turn: Turn) -> dict:
        if not turn.tool_calls:
            return {"role": "assistant", "content": turn.text}
        # transformers wants `arguments` as a dict here, unlike the OpenAI API
        # which uses a JSON string. Passing a string confuses many templates.
        return {
            "role": "assistant",
            "content": turn.text,
            "tool_calls": [
                {"type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                for c in turn.tool_calls
            ],
        }

    def add_tool_results(self, results: list[ToolResult]) -> None:
        for r in results:
            self.messages.append({"role": "tool", "name": r.name, "content": r.content})
