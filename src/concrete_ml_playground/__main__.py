import click
import tempfile
import os

from datetime import datetime

from . import logger
from .dataset_collector import get_dataset_loaders
from .experiment_collector import get_inference_experiments, get_training_experiments
from .statistics_handler import evaluate_experiment_results
from .draw_plots import (
    draw_dataset,
    draw_decision_boundary,
    draw_feature_dim_runtime_plot,
    redraw_decision_boundary,
)
from .csv_handler import init_csv, append_result_to_csv


@click.command()
@click.option("--all_exps", is_flag=True, help="Run all training and inference experiments")
@click.option("--all_inference_exps", is_flag=True, help="Run only the inference experiments")
@click.option(
    "--exp", type=str, required=False, multiple=True, help="Run only the specified experiment(s)"
)
@click.option("--all_dsets", is_flag=True, help="Run on all datasets")
@click.option(
    "--dset", type=str, required=False, multiple=True, help="Run only on the specified dataset(s)"
)
@click.option(
    "--draw_all", is_flag=True, help="Draw all plots in addition to running the experiments"
)
@click.option(
    "--draw_cheap",
    is_flag=True,
    help="Draw the computationally cheap plots (i.e. not decision boundaries) in addition to running the experiments",
)
@click.option(
    "--redraw",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, readable=True, writable=True, resolve_path=True
    ),
    required=False,
    help="Redraw all graphs in the supplied results directory without running any experiments or re-doing expensive computations. Useful for stylistic changes in existing plots",
)
@click.option(
    "--execs",
    type=int,
    default=1,
    show_default=True,
    help="How often each experiment should run. Will calculate the mean value and standard deviation between all executions",
)
def main(
    all_exps: bool,
    all_inference_exps: bool,
    exp: list[str],
    all_dsets: bool,
    dset: list[str],
    execs: int,
    draw_all: bool,
    draw_cheap: bool,
    redraw: click.Path | None,
):
    dataset_loaders = get_dataset_loaders()
    scheduled_dataset_loaders = {}
    if all_dsets:
        scheduled_dataset_loaders = dataset_loaders
    elif len(dset) > 0:
        for ds in dset:
            dset_loader = dataset_loaders.get(ds)
            if dset_loader is not None:
                scheduled_dataset_loaders[ds] = dset_loader
            else:
                possible_values = list(dataset_loaders.keys())
                raise Exception(
                    f"No dataset with name '{ds}' exists. --dset can only have the following values: {possible_values}."
                )
    else:
        raise Exception("Either --all_dsets or --dset option is required")

    scheduled_exps = {}

    inf_exp_loaders = get_inference_experiments()
    train_exp_loaders = get_training_experiments()
    if all_exps:
        scheduled_exps = {**inf_exp_loaders, **train_exp_loaders}
    elif all_inference_exps:
        scheduled_exps = inf_exp_loaders
    elif len(exp) > 0:
        for ex in exp:
            train_exp = train_exp_loaders.get(ex)
            inf_exp = inf_exp_loaders.get(ex)
            if train_exp is not None:
                scheduled_exps[ex] = train_exp
            elif inf_exp is not None:
                scheduled_exps[ex] = inf_exp
            else:
                possible_values = list(inf_exp_loaders.keys()) + list(train_exp_loaders.keys())
                raise Exception(
                    f"No experiment with name '{ex}' exists. --exp can only have the following values: {possible_values}."
                )
    else:
        raise Exception("Either --all_exps, --all_inference_exps or --exp option is required")

    if redraw is None:
        timestamp = datetime.now().isoformat()
        results_dir = f"results_{timestamp}"
        if os.path.exists(results_dir):
            raise Exception(f"The directory '{results_dir}' already exists!")
        os.makedirs(results_dir)
        logger.info(f"Successfully created results directory '{results_dir}'")
        init_csv(results_dir)
    else:
        results_dir = str(redraw)

    for dset_name_dict, (dset_loader, dset_name) in scheduled_dataset_loaders.items():
        X_train, X_test, y_train, y_test = dset_loader()
        if draw_all or draw_cheap or redraw:
            draw_dataset(results_dir, dset_name, X_train, X_test, y_train, y_test)
        for exp_name_dict, (exp_func, exp_name) in scheduled_exps.items():
            if redraw is not None:
                redraw_decision_boundary(results_dir, exp_name, dset_name, X_test, y_test)
            else:
                results = []
                for i in range(execs):
                    logger.info(
                        f"Running '{exp_name}' experiment on '{dset_name}' dataset [{i + 1} of {execs}]..."
                    )
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        result, plot_data = exp_func(tmpdirname, X_train, X_test, y_train, y_test)
                    results.append(result)
                final_result = evaluate_experiment_results(
                    results, dset_name, dset_name_dict, exp_name, exp_name_dict
                )
                logger.info(
                    f"Mean result of {execs} executions of '{exp_name}' experiment on '{dset_name}' dataset: {final_result}"
                )
                logger.info(
                    f"The main processing with FHE was {final_result.fhe_duration_processing / final_result.clear_duration} times slower than normal processing on clear data"
                )
                append_result_to_csv(results_dir, final_result)
                if draw_all:
                    draw_decision_boundary(
                        results_dir, plot_data, exp_name, dset_name, X_test, y_test
                    )

    if (all_exps or all_inference_exps) and (draw_all or draw_cheap or redraw):
        draw_feature_dim_runtime_plot(results_dir, "synth_")
        draw_feature_dim_runtime_plot(results_dir, "spam_")


if __name__ == "__main__":
    main()
