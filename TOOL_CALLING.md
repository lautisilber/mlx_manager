# Tool-calling: implementation plan (not yet built)

Status: **not implemented**. This document captures the design/scope analysis done
before deciding whether to build it, so it doesn't need to be re-derived later.
`pack_serve.py` currently ignores any `tools` field in a request and never returns
`tool_calls` — Agent-mode-style clients (VS Code Agent mode, etc.) will get plain text
back, not real tool invocations.

## Why it's not just a format reshape

Unlike streaming, `/v1/completions`, or per-request sampling overrides (all format
translations of data we already had), tool-calling needs:
1. A real parser for the model's tag-based function-call output (not JSON).
2. A small state machine for streaming (a tool call isn't valid until it's fully generated).
3. One input-side transform for how tool-call arguments are represented.

It's still possible — the base model has tool-calling trained in, and the pack's own
`chat_template.jinja` fully supports it — but it's a meaningfully bigger and riskier
change than anything built so far. Rough estimate: **~150-200 new lines**, split
roughly 100 in `pack_common.py` (nearly doubling its current ~200 lines) and 60-70 in
`pack_serve.py`. `pack_chat.py` needs no changes — it has no tool-execution loop, so
tool-calling is inherently a `pack_serve.py`/`pack_common.py`-only feature.

## What the chat template actually expects

Read in full from `<snapshot>/chat_template.jinja` (169 lines) for the Bonsai pack:

- **`tools`**: a Jinja variable, iterable of tool definitions, rendered verbatim via
  `tool | tojson` into a system-message block (template lines 57-75). Passing
  `tools=<list of OpenAI-style tool schema dicts>` into the `.render(...)` call should
  just work — no transformation needed for the tool *definitions* themselves.
- **Tool results (`role: "tool"` messages)**: handled natively by the template
  (lines 147-158) — it wraps `message.content` in `<tool_response>...</tool_response>`
  and groups consecutive tool messages under one `<|im_start|>user` block itself.
  **No pre-wrapping needed on our side.** (This was flagged as an open risk in an
  earlier pass over just a grep excerpt of the template; reading the full file resolved
  it — turned out not to be a problem.)
- **Assistant messages with `tool_calls`**: the template does
  `tool_call.arguments|items` (line 136) — i.e. it expects `arguments` as an actual
  dict/mapping. OpenAI's wire format sends `tool_calls[].function.arguments` as a
  **JSON string**. This is the one real input-side transform needed: `json.loads()`
  each `arguments` string before rendering, for any assistant message with tool calls
  in the conversation history.
- **Output format**: the model is instructed (and expected) to reply with
  ```
  <tool_call>
  <function=name>
  <parameter=arg_name>
  value
  </parameter>
  </function>
  </tool_call>
  ```
  Parameter values can span multiple lines (per the template's own instruction text at
  line 68). Reasoning, if any, comes *before* the tool call, never after.
- **Reasoning (`<think>`) is already handled and not part of this scope**: the
  template always wraps assistant output in `<think>...</think>`, but
  `pack_common.render()` already passes `enable_thinking=False`, which makes the
  template inject an already-closed empty think block into the *prompt* (template
  lines 165-169), suppressing visible reasoning. This is why current responses come
  back clean. No change needed here for tool-calling.

## Planned changes

### `pack_common.py`
- `render(pack, messages, tools=None)` — add `tools` param, pass through to
  `.render(messages=messages, tools=tools, ...)`. ~2 lines.
- `normalize_tool_call_messages(messages)` — deep-copy messages, `json.loads()` any
  string `tool_calls[].function.arguments` into a dict. ~15 lines.
- `parse_tool_calls(text) -> (visible_content, tool_calls | None)` — the actual new
  logic. Extract `<tool_call>...</tool_call>` blocks (regex with `re.DOTALL`, since
  parameter values can be multi-line), extract `<parameter=x>...</parameter>` pairs
  within each, build OpenAI-shaped `tool_calls` entries with `arguments` re-serialized
  as a JSON *string* (the reverse direction from the input-side transform).
  `visible_content` is everything before the first `<tool_call>` tag. ~40-50 lines.
- `ToolCallBuffer` (or similar) — a small streaming-state helper, conceptually similar
  to the existing `StopMatcher` but for a start/end tag pair instead of a fixed string:
  stream normal text through as usual, but the moment `<tool_call>` appears, stop
  emitting content deltas and buffer silently until `</tool_call>` closes. ~30 lines.

### `pack_serve.py`
- Extract `tools` from the request body (and decide whether to support `tool_choice`
  forcing — probably skip forcing semantics initially and only support the default
  "auto" behavior, to keep scope down).
- Call `normalize_tool_call_messages()` before `render()`, pass `tools` through.
- Non-streaming: after `generate()`, run `parse_tool_calls()`; if tool calls were
  found, shape the response as `{"role": "assistant", "content": None or the
  pre-call text, "tool_calls": [...]}` with `finish_reason: "tool_calls"` instead of
  `"stop"`/`"length"`. ~20 lines.
- Streaming: wire in `ToolCallBuffer` — normal text still streams token-by-token;
  once a tool-call region is detected, buffer until complete, then emit a single
  `delta.tool_calls` chunk. ~30-40 lines.

## Real risk that isn't just a coding problem

Whether the model reliably produces well-formed tags for realistic *multi-tool*
schemas is a model-behavior question, not a correctness question — needs actual
testing with a real tool definition (e.g. a fake `get_weather` or `read_file` tool)
to see how it holds up before trusting the parser in practice, and likely some
hardening afterward for malformed/partial output.

## Non-goals (for a first pass, if built)

- `tool_choice` forcing (requiring a specific tool, or `"none"`) — support only the
  default "auto" behavior initially.
- `pack_chat.py` — no interactive tool-execution loop exists there; out of scope.
- Parallel tool calls are structurally supported by the parser design (multiple
  `<tool_call>` blocks) but haven't been thought through for streaming ordering.
