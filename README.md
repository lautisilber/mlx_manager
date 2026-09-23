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

Copies `mlx` to `~/.local/bin/mlx` (make sure `~/.local/bin` is on your `PATH`). Run `make uninstall` to remove it. Set `PREFIX` to install elsewhere, e.g. `make install PREFIX=/usr/local`.

### Tab completion (optional)

Tab completion for subcommands and installed model names (e.g. `mlx serve mlx-com<TAB>`) is entirely opt-in. To enable it, add one line to your shell's rc file (`~/.zshrc` or `~/.bashrc`):

```sh
source <(mlx completion)
```

`mlx completion` prints shell completion code that auto-detects bash vs. zsh when sourced (bootstrapping `compinit`/`bashcompinit` for zsh if they aren't already loaded). If you never add that line, nothing about completion is touched. Requires `mlx` to be on your `PATH`.
