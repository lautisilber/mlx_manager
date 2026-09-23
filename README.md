# MLX Manager

A bash script tool that allows the user to manage local AI models from the terminal, wrapping Apple's mlx-lm. It lets the user manage the local models with a few commands

- ```mlx create-venv``` Creates the python venv used to run mlx-lm
- ```mlx delete-venv``` Deletes the python venv used to run mlx-lm
- ```mlx get <huggingface name>``` Lets the user get the desired model from huggingface
- ```mlx list``` Lists all installed models
- ```mlx chat <huggingface name>``` Lets the user chat in the window with the desired model. If the model isn't installed, it lets the user know how to do so with ```mlx get```
- ```mlx serve <huggingface name> <port>``` Lets the user serve the desired model in the specified port
- ```mlx delete <huggingface name>``` Lets the user delete an installed model

## Setup

```sh
make install
```

Copies `mlx` to `~/.local/bin`, and the `mlx_manager/` support folder (containing the `pack_*.py` helpers) to `~/.local/bin/mlx_manager/` (make sure `~/.local/bin` is on your `PATH`). Run `make uninstall` to remove them. Set `PREFIX` to install elsewhere, e.g. `make install PREFIX=/usr/local`.

### Venvs

There are two venvs, and you only ever directly manage one of them:

- **`~/.mlx-manager/venv`** — the normal venv, used for everything by default (`mlx get`, `mlx chat`, `mlx serve` on any regular model). `mlx create-venv` always creates this one (via `python3 -m venv`, so it still picks up whatever Python `pyenv` or similar has active on your `PATH`) and installs `mlx-lm` into it. This is the venv that must exist before you can use the tool at all.
- **`~/.mlx-manager/venv-runtime`** — a second venv, only for models whose pack declares `requires_runtime` in `config.json` (packs that ship their own loader because their architecture or quantization scheme isn't supported by stock mlx-lm — see below). You never create this yourself: `mlx chat` detects such a pack automatically and creates this venv the first time one is needed, installing that pack's `runtime/requirements.txt` into it. Keeping it separate means a pack's pinned dependency versions (which can differ a lot from what `mlx-lm` normally uses) never affect the normal venv.

`mlx delete-venv` removes both venvs if present, in one confirmation — you don't need to track them separately when cleaning up.

Some HuggingFace packs (e.g. novel quantization schemes) ship their own loader in a `runtime/` folder instead of relying on mlx-lm's architecture registry. When `mlx chat` detects one (via the `venv-runtime` venv above), it chats via `mlx_manager/pack_chat.py`'s REPL instead of `mlx_lm.chat`. `mlx serve` does not yet support these packs.

### Tab completion (optional)

Tab completion for subcommands and installed model names (e.g. `mlx serve mlx-com<TAB>`) is entirely opt-in. To enable it, add one line to your shell's rc file (`~/.zshrc` or `~/.bashrc`):

```sh
source <(mlx completion)
```

`mlx completion` prints shell completion code that auto-detects bash vs. zsh when sourced (bootstrapping `compinit`/`bashcompinit` for zsh if they aren't already loaded). If you never add that line, nothing about completion is touched. Requires `mlx` to be on your `PATH`.
