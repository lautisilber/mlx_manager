#!/usr/bin/env python3
"""Interactive text chat loop for MLX model packs that ship their own runtime loader.

Some packs (e.g. custom low-bit quantization schemes) declare an architecture
mlx-lm doesn't know about and bundle a loader in runtime/ instead (config.json
sets "requires_runtime"). This mirrors mlx_lm.chat's REPL for such packs, using
the pack's own runtime/artifact.py loader.

Some of these packs are saved as a full vision-language model even when you
only want text chat (their weights are stored under a "language_model."
namespace that the plain text loader rejects), so runtime/vision_artifact.py
and mlx-vlm are used to load them instead. No image input is supported either
way; this is only about which loader can read the pack's weights.
"""

import json
import sys
from pathlib import Path

PACK = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(PACK / "runtime"))

MAX_TOKENS = 2048


def read_json(name, default):
    path = PACK / name
    return json.loads(path.read_text()) if path.is_file() else default


def sampler_settings():
    gen = read_json("generation_config.json", {})
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


def render(messages):
    template_path = PACK / "chat_template.jinja"
    if not template_path.is_file():
        return messages[-1]["content"]
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    return (
        ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        .from_string(template_path.read_text())
        .render(messages=messages, add_generation_prompt=True, enable_thinking=False)
    )


def pack_needs_vlm_loader():
    """True if this pack's weights are stored as a full VLM and need mlx-vlm to load,
    even though we only ever generate text (no image input is offered)."""
    return bool(read_json("config.json", {}).get("components", {}).get("vision"))


def make_text_generator():
    import mlx.core as mx
    from artifact import load_model
    from mlx_lm.sample_utils import make_logits_processors, make_sampler
    from tokenizers import Tokenizer

    model, _ = load_model(PACK)
    tokenizer = Tokenizer.from_file(str(PACK / "tokenizer.json"))
    eos_cfg = read_json("generation_config.json", {}).get("eos_token_id")
    stop = set(eos_cfg if isinstance(eos_cfg, list) else [] if eos_cfg is None else [eos_cfg])
    stop |= {
        i
        for i in (tokenizer.token_to_id("<|im_end|>"), tokenizer.token_to_id("<|endoftext|>"))
        if i is not None
    }
    settings = sampler_settings()
    sample = make_sampler(
        temp=settings["temperature"],
        top_p=settings["top_p"],
        top_k=settings["top_k"],
        min_p=settings.get("min_p", 0.0),
    )

    def generate(messages):
        processors = make_logits_processors(repetition_penalty=settings.get("repetition_penalty"))
        x = mx.array([tokenizer.encode(render(messages), add_special_tokens=False).ids])
        cache = model.make_cache()
        generated = []
        for _ in range(MAX_TOKENS):
            logits = model.lm_head(model.model(x, cache=cache)[:, -1:, :])[:, -1, :]
            for processor in processors:
                logits = processor(mx.array(generated), logits)
            x = sample(logits)[:, None]
            mx.eval(x)
            token = int(x.item())
            if token in stop:
                break
            generated.append(token)
            print(tokenizer.decode([token]), end="", flush=True)
        print()
        return tokenizer.decode(generated)

    return generate


def make_vlm_text_generator():
    """Text-only generation through mlx-vlm, for packs whose weights need that loader."""
    from mlx_vlm import generate as vlm_generate
    from vision_artifact import load_vl_model

    model, processor, _ = load_vl_model(PACK)
    settings = sampler_settings()

    def generate(messages):
        result = vlm_generate(model, processor, render(messages), max_tokens=MAX_TOKENS, **settings)
        text = result if isinstance(result, str) else result.text
        print(text)
        return text

    return generate


def main():
    print(f"[INFO] Starting chat session with custom runtime pack at {PACK}")
    print("The command list:\n- 'q' to exit\n- 'r' to reset the chat\n- 'h' to display these commands")
    generate = make_vlm_text_generator() if pack_needs_vlm_loader() else make_text_generator()
    messages = []
    while True:
        try:
            query = input(">> ")
        except EOFError:
            break
        if query == "q":
            break
        if query == "r":
            messages = []
            continue
        if query == "h":
            print("- 'q' to exit\n- 'r' to reset the chat\n- 'h' to display these commands")
            continue
        messages.append({"role": "user", "content": query})
        reply = generate(messages)
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
