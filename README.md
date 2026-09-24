# MLX Manager

A bash script tool that allows the user to manage local AI models from the terminal, wrapping Apple's mlx-lm. It lets the user manage the local models with a few commands

- ```mlx create-venv``` Creates the python venv used to run mlx-lm (optional — created automatically on first use if you skip this)
- ```mlx delete-venv``` Deletes the python venv used to run mlx-lm
- ```mlx get <huggingface name>``` Lets the user get the desired model from huggingface
- ```mlx list``` Lists all installed models
- ```mlx chat <huggingface name>``` Lets the user chat in the window with the desired model. If the model isn't installed, it lets the user know how to do so with ```mlx get```
- ```mlx serve <huggingface name> [--port <port>] [--host <host>]``` Lets the user serve the desired model (defaults: port 8080, host 127.0.0.1)
- ```mlx delete <huggingface name>``` Lets the user delete an installed model

## Setup

```sh
make install
```

Copies `mlx` to `~/.local/bin`, and the `mlx_manager/` support folder (containing the `pack_*.py` helpers) to `~/.local/bin/mlx_manager/` (make sure `~/.local/bin` is on your `PATH`). Run `make uninstall` to remove them. Set `PREFIX` to install elsewhere, e.g. `make install PREFIX=/usr/local`.

### Venvs

There are two venvs, and you only ever directly manage one of them:

- **`~/.mlx-manager/venv`** — the normal venv, used for everything by default (`mlx get`, `mlx chat`, `mlx serve` on any regular model). You don't need to create this yourself: it's created lazily the first time any command needs it (via `python3 -m venv`, so it still picks up whatever Python `pyenv` or similar has active on your `PATH`), installing `mlx-lm` into it, then reused for every normal model after that. `mlx create-venv` still exists if you'd rather provision it explicitly ahead of time (e.g. to control which Python it's built from); it's a no-op if the venv already exists.
- **`~/.mlx-manager/venv-runtime`** — a second venv, only for models whose pack declares `requires_runtime` in `config.json` (packs that ship their own loader because their architecture or quantization scheme isn't supported by stock mlx-lm — see below). You never create this yourself: `mlx chat`/`mlx serve` detect such a pack automatically and create this venv the first time one is needed, installing that pack's `runtime/requirements.txt` into it. Keeping it separate means a pack's pinned dependency versions (which can differ a lot from what `mlx-lm` normally uses) never affect the normal venv.

`mlx delete-venv` removes both venvs if present, in one confirmation — you don't need to track them separately when cleaning up.

### Custom-runtime packs

Some HuggingFace packs (e.g. novel quantization schemes) ship their own loader in a `runtime/` folder instead of relying on mlx-lm's architecture registry. When `mlx chat`/`mlx serve` detect one (via the `venv-runtime` venv above), they use `mlx_manager/pack_chat.py` (a REPL, in place of `mlx_lm.chat`) or `mlx_manager/pack_serve.py` (a minimal OpenAI-compatible `/v1/chat/completions` + `/v1/models` server, in place of `mlx_lm.server`) instead. Both share their model-loading/generation logic via `mlx_manager/pack_common.py`.

`pack_serve.py` supports both a single-JSON-body response and, when the client sets `"stream": true`, a real Server-Sent-Events stream of `chat.completion.chunk` objects (HTTP/1.1 chunked transfer encoding), so it should work with clients that require streaming as well as ones that don't.

### Tab completion (optional)

Tab completion for subcommands and installed model names (e.g. `mlx serve mlx-com<TAB>`) is entirely opt-in. To enable it, add one line to your shell's rc file (`~/.zshrc` or `~/.bashrc`):

```sh
source <(mlx completion)
```

`mlx completion` prints shell completion code that auto-detects bash vs. zsh when sourced (bootstrapping `compinit`/`bashcompinit` for zsh if they aren't already loaded). If you never add that line, nothing about completion is touched. Requires `mlx` to be on your `PATH`.
