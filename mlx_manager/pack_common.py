"""Shared model-loading and generation helpers for custom-runtime packs.

Some packs (e.g. custom low-bit quantization schemes) declare an architecture
mlx-lm doesn't know about and bundle a loader in runtime/ instead (config.json
sets "requires_runtime"). Used by pack_chat.py (interactive REPL) and
pack_serve.py (OpenAI-compatible server), both of which use the pack's own
runtime/artifact.py loader.

Some of these packs are saved as a full vision-language model even when only
text is wanted (their weights are stored under a "language_model." namespace
that the plain text loader rejects), so runtime/vision_artifact.py and
mlx-vlm are used to load them instead. No image input is supported either
way; this is only about which loader can read the pack's weights.
"""

import json
import sys

MAX_TOKENS = 2048


def read_json(pack, name, default):
    path = pack / name
    return json.loads(path.read_text()) if path.is_file() else default


def sampler_settings(pack):
    gen = read_json(pack, "generation_config.json", {})
    out = {
        "temperature": gen.get("temperature", 1.0),
        "top_p": gen.get("top_p", 0.95),
        "top_k": gen.get("top_k", 20),
    }
    if gen.get("do_sample") is False:
        out["temperature"] = 0.0
    if gen.get("min_p"):
        out["min_p"] = gen["min_p"]
    if gen.get("repetition_penalty") not in (None, 1.0):
        out["repetition_penalty"] = gen["repetition_penalty"]
    return out


def render(pack, messages):
    template_path = pack / "chat_template.jinja"
    if not template_path.is_file():
        return messages[-1]["content"]
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    return (
        ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        .from_string(template_path.read_text())
        .render(messages=messages, add_generation_prompt=True, enable_thinking=False)
    )


def pack_needs_vlm_loader(pack):
    """True if this pack's weights are stored as a full VLM and need mlx-vlm to load,
    even though we only ever generate text (no image input is offered)."""
    return bool(read_json(pack, "config.json", {}).get("components", {}).get("vision"))


def make_text_generator(pack, max_tokens=MAX_TOKENS):
    sys.path.insert(0, str(pack / "runtime"))
    import mlx.core as mx
    from artifact import load_model
    from mlx_lm.sample_utils import make_logits_processors, make_sampler
    from tokenizers import Tokenizer

    model, _ = load_model(pack)
    tokenizer = Tokenizer.from_file(str(pack / "tokenizer.json"))
    eos_cfg = read_json(pack, "generation_config.json", {}).get("eos_token_id")
    stop = set(eos_cfg if isinstance(eos_cfg, list) else [] if eos_cfg is None else [eos_cfg])
    stop |= {
        i
        for i in (tokenizer.token_to_id("<|im_end|>"), tokenizer.token_to_id("<|endoftext|>"))
        if i is not None
    }
    defaults = sampler_settings(pack)

    def generate(
        prompt_text,
        on_token=None,
        max_tokens=max_tokens,
        temperature=None,
        top_p=None,
        top_k=None,
        min_p=None,
        repetition_penalty=None,
    ):
        sample = make_sampler(
            temp=temperature if temperature is not None else defaults["temperature"],
            top_p=top_p if top_p is not None else defaults["top_p"],
            top_k=top_k if top_k is not None else defaults["top_k"],
            min_p=min_p if min_p is not None else defaults.get("min_p", 0.0),
        )
        processors = make_logits_processors(
            repetition_penalty=(
                repetition_penalty if repetition_penalty is not None else defaults.get("repetition_penalty")
            )
        )
        x = mx.array([tokenizer.encode(prompt_text, add_special_tokens=False).ids])
        cache = model.make_cache()
        generated = []
        finish_reason = "length"
        for _ in range(max_tokens):
            logits = model.lm_head(model.model(x, cache=cache)[:, -1:, :])[:, -1, :]
            for processor in processors:
                logits = processor(mx.array(generated), logits)
            x = sample(logits)[:, None]
            mx.eval(x)
            token = int(x.item())
            if token in stop:
                finish_reason = "stop"
                break
            generated.append(token)
            if on_token:
                on_token(tokenizer.decode([token]))
        return tokenizer.decode(generated), len(generated), finish_reason

    return generate


def make_vlm_text_generator(pack, max_tokens=MAX_TOKENS):
    """Text-only generation through mlx-vlm, for packs whose weights need that loader."""
    sys.path.insert(0, str(pack / "runtime"))
    from mlx_vlm import stream_generate
    from vision_artifact import load_vl_model

    model, processor, _ = load_vl_model(pack)
    defaults = sampler_settings(pack)

    def generate(
        prompt_text,
        on_token=None,
        max_tokens=max_tokens,
        temperature=None,
        top_p=None,
        top_k=None,
        min_p=None,
        repetition_penalty=None,
    ):
        settings = dict(defaults)
        overrides = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
        }
        settings.update({k: v for k, v in overrides.items() if v is not None})

        chunks = []
        completion_tokens = 0
        finish_reason = "length"
        for result in stream_generate(
            model, processor, prompt_text, max_tokens=max_tokens, **settings
        ):
            if result.text:
                chunks.append(result.text)
                if on_token:
                    on_token(result.text)
            completion_tokens = result.generation_tokens
            if result.finish_reason:
                finish_reason = result.finish_reason
        return "".join(chunks), completion_tokens, finish_reason

    return generate


def make_generator(pack, max_tokens=MAX_TOKENS):
    """Returns generate(prompt_text, on_token=None, max_tokens=None, temperature=None,
    top_p=None, top_k=None, min_p=None, repetition_penalty=None)
    -> (text, completion_token_count, finish_reason), using whichever loader this pack's
    weights actually need. prompt_text is fed to the tokenizer as-is; callers that have a
    chat message list should render() it first. Any sampling kwarg left as None falls back
    to the pack's own generation_config.json default."""
    factory = make_vlm_text_generator if pack_needs_vlm_loader(pack) else make_text_generator
    return factory(pack, max_tokens)
