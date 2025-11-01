import click
import tempfile
import os

from datetime import datetime

from concrete_ml_playground.interfaces import ExperimentOutput

from . import logger
from .dataset_collector import get_dataset_loaders, get_ner_dataset_loaders
from .experiment_collector import (
    get_inference_experiments,
    get_training_experiments,
    get_ner_experiments,
)
from .statistics_handler import (
    experiment_output_processor,
    evaluate_experiment_results,
)
from .draw_plots import (
    draw_dataset,
    draw_decision_boundary,
    draw_feature_dim_runtime_plot,
    redraw_decision_boundary,
)
from .csv_handler import init_csv, append_result_to_csv, read_csv


@click.command()
@click.option("--all_exps", is_flag=True, help="Run all types of experiments")
@click.option("--all_inference_exps", is_flag=True, help="Run the inference experiments")
@click.option("--all_training_exps", is_flag=True, help="Run the training experiments")
@click.option("--all_ner_exps", is_flag=True, help="Run the NER experiments")
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
    "--resume",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, readable=True, writable=True, resolve_path=True
    ),
    required=False,
    help="Resume interrupted experiment run by only executing experiment/dataset combinations that are not in the csv of the supplied result directory already",
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
    all_training_exps: bool,
    all_ner_exps: bool,
    exp: list[str],
    all_dsets: bool,
    dset: list[str],
    execs: int,
    draw_all: bool,
    draw_cheap: bool,
    redraw: click.Path | None,
    resume: click.Path | None,
):
    # NER experiments are separate from the rest because they can only run on NER datasets
    dataset_loaders = get_dataset_loaders()
    ner_dset_loaders = get_ner_dataset_loaders()
    scheduled_dataset_loaders = {}
    scheduled_ner_dataset_loaders = {}
    if all_dsets:
        scheduled_dataset_loaders = dataset_loaders
        scheduled_ner_dataset_loaders = ner_dset_loaders
    elif len(dset) > 0:
        for ds in dset:
            dset_loader = dataset_loaders.get(ds)
            ner_dset_loader = ner_dset_loaders.get(ds)
            if dset_loader is not None:
                scheduled_dataset_loaders[ds] = dset_loader
            elif ner_dset_loader is not None:
                scheduled_ner_dataset_loaders[ds] = ner_dset_loader
            else:
                possible_values = list(dataset_loaders.keys()) + list(ner_dset_loaders.keys())
                raise Exception(
                    f"No dataset with name '{ds}' exists. --dset can only have the following values: {possible_values}."
                )
    else:
        raise Exception("Either --all_dsets or --dset option is required")

    inf_exp_loaders = get_inference_experiments()
    train_exp_loaders = get_training_experiments()
    ner_exp_loaders = get_ner_experiments()
    scheduled_exps = {}
    scheduled_ner_exps = {}
    option_exists = False
    if all_exps:
        scheduled_exps = {**inf_exp_loaders, **train_exp_loaders}
        scheduled_ner_exps = ner_exp_loaders
        option_exists = True
    else:
        if all_inference_exps:
            scheduled_exps = inf_exp_loaders
            option_exists = True
        if all_training_exps:
            scheduled_exps = {**scheduled_exps, **train_exp_loaders}
            option_exists = True
        if all_ner_exps:
            scheduled_ner_exps = ner_exp_loaders
            option_exists = True
        if not option_exists and len(exp) > 0:
            for ex in exp:
                train_exp = train_exp_loaders.get(ex)
                inf_exp = inf_exp_loaders.get(ex)
                ner_exp = ner_exp_loaders.get(ex)
                if train_exp is not None:
                    scheduled_exps[ex] = train_exp
                elif inf_exp is not None:
                    scheduled_exps[ex] = inf_exp
                elif ner_exp is not None:
                    scheduled_ner_exps[ex] = ner_exp
                else:
                    possible_values = (
                        list(inf_exp_loaders.keys())
                        + list(train_exp_loaders.keys())
                        + list(scheduled_ner_exps)
                    )
                    raise Exception(
                        f"No experiment with name '{ex}' exists. --exp can only have the following values: {possible_values}."
                    )
            option_exists = True
    if not option_exists:
        raise Exception(
            "Either --all_exps, --all_inference_exps, --all_training_exps, all_ner_exps, or --exp option is required"
        )

    if redraw is not None:
        results_dir = str(redraw)
    elif resume is not None:
        results_dir = str(resume)
    else:
        timestamp = datetime.now().isoformat()
        results_dir = f"results_{timestamp}"
        if os.path.exists(results_dir):
            raise Exception(f"The directory '{results_dir}' already exists!")
        os.makedirs(results_dir)
        logger.info(f"Successfully created results directory '{results_dir}'")
        init_csv(results_dir)

    if resume is not None:
        done_exps = read_csv(str(resume))
    else:
        done_exps = []

    for ner_dset_name_dict, (
        ner_dset_loader,
        ner_dset_name,
    ) in scheduled_ner_dataset_loaders.items():
        dset_info = ner_dset_loader()
        for ner_exp_name_dict, (ner_exp_func, ner_exp_name) in scheduled_ner_exps.items():
            skip = False
            for done_exp in done_exps:
                if (
                    ner_exp_name_dict == done_exp.exp_name_dict
                    and ner_dset_name_dict == done_exp.dset_name_dict
                ):
                    skip = True
                    break
            if not skip:
                results = []
                exp_out: ExperimentOutput
                for i in range(execs):
                    logger.info(
                        f"Running '{ner_exp_name}' experiment on '{ner_dset_name}' dataset [{i + 1} of {execs}]..."
                    )
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        exp_out = ner_exp_func(tmpdirname, dset_info)
                        result = experiment_output_processor(dset_info.y_test, exp_out)
                    results.append(result)
                final_result = evaluate_experiment_results(
                    results, ner_dset_name, ner_dset_name_dict, ner_exp_name, ner_exp_name_dict
                )
                logger.info(
                    f"Mean result of {execs} executions of '{ner_exp_name}' experiment on '{ner_dset_name}' dataset: {final_result}"
                )
                logger.info(
                    f"The main processing with FHE was {final_result.fhe_duration_processing / final_result.clear_duration} times slower than normal processing on clear data"
                )
                append_result_to_csv(results_dir, final_result)
            else:
                logger.info(
                    f"Found existing '{ner_exp_name}' experiment on '{ner_dset_name}' dataset in results, skipping..."
                )

    for dset_name_dict, (dset_loader, dset_name) in scheduled_dataset_loaders.items():
        X_train, X_test, y_train, y_test = dset_loader()
        if draw_all or draw_cheap or redraw:
            draw_dataset(results_dir, dset_name, X_train, X_test, y_train, y_test)
        for exp_name_dict, (exp_func, exp_name) in scheduled_exps.items():
            skip = False
            for done_exp in done_exps:
                if (
                    exp_name_dict == done_exp.exp_name_dict
                    and dset_name_dict == done_exp.dset_name_dict
                ):
                    skip = True
                    break
            if redraw is not None:
                redraw_decision_boundary(results_dir, exp_name, dset_name, X_test, y_test)
            elif not skip:
                results = []
                exp_out: ExperimentOutput
                for i in range(execs):
                    logger.info(
                        f"Running '{exp_name}' experiment on '{dset_name}' dataset [{i + 1} of {execs}]..."
                    )
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        exp_out = exp_func(tmpdirname, X_train, X_test, y_train)
                        result = experiment_output_processor(y_test, exp_out)
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
                        results_dir, exp_out, exp_name, dset_name, X_test, y_test
                    )
            else:
                logger.info(
                    f"Found existing '{exp_name}' experiment on '{dset_name}' dataset in results, skipping..."
                )

    if (all_exps or all_inference_exps) and (draw_all or draw_cheap or redraw):
        draw_feature_dim_runtime_plot(results_dir, "synth_")
        draw_feature_dim_runtime_plot(results_dir, "spam_")


if __name__ == "__main__":
    main()
