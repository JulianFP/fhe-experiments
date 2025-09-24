# Experiments and the LaTex document of my Bachelor Thesis

## Install dependencies

First you have to have `uv` installed. Either enter the nix shell by running `nix develop` or refer to [the UV docs](https://docs.astral.sh/uv/getting-started/installation/) for installing it without Nix.

After that you can just run

`uv sync`

## Run experiments

To run the full experiment suite while drawing all graphs execute:

`uv run concrete_ml_playground --all_exps --all_dsets --draw_all --execs 10`

This will take forever though. A more sensible but still quite exhaustive execution could be:

`uv run concrete_ml_playground --all_inference_exps --all_dsets --draw_cheap --execs 10`

You can also choose to just run specific experiments/datasets. The following command will execute the `log_reg` and `neural_net` experiments on both the `xor` and `spam_50` datasets:

`uv run concrete_ml_playground --exp log_reg --exp neural_net --dset xor --dset spam_50 --execs 10`

To redraw graphs of existing results (e.g. to change their styling) execute:

`uv run concrete_ml_playground --redraw <your results directory> --all_dsets --all_exps`

Use the following command to see all available arguments:

`uv run concrete_ml_playground --help`
