import click
import shutil
from sklearn.model_selection import train_test_split

from .dataset_collector import get_dataset_loaders
from .experiment_collector import get_inference_experiments, get_training_experiments
from .statistics_handler import evaluate_experiment_results
from .draw_plots import (
    draw_decision_boundary,
    draw_feature_dim_runtime_plot,
    redraw_decision_boundary,
)
from .csv_handler import write_result_to_csv


@click.command()
@click.option("--all_exps", is_flag=True, help="Run all training and inference experiments")
@click.option("--all_inference_exps", is_flag=True, help="Run only the inference experiments")
@click.option("--exp", type=str, required=False, help="Run only the specified experiment")
@click.option("--all_dsets", is_flag=True, help="Run on all datasets")
@click.option("--dset", type=str, required=False, help="Run only on the specified dataset")
@click.option("--draw", is_flag=True, help="Draw plots after running the experiments")
@click.option(
    "--redraw",
    is_flag=True,
    help="Redraw existing plots without running any experiments. Useful for stylistic changes in the plots",
)
@click.option(
    "--execs",
    type=int,
    default=1,
    show_default=True,
    help="How often each experiment should run. Will calculate the mean value between all executions",
)
def main(
    all_exps: bool,
    all_inference_exps: bool,
    exp: str | None,
    all_dsets: bool,
    dset: str | None,
    execs: int,
    draw: bool,
    redraw: bool,
):
    dataset_loaders = get_dataset_loaders()
    scheduled_dataset_loaders = {}
    if all_dsets:
        scheduled_dataset_loaders = dataset_loaders
    elif dset is not None:
        dset_loader = dataset_loaders.get(dset)
        if dset_loader is not None:
            scheduled_dataset_loaders[dset] = dset_loader
        else:
            raise Exception(f"No dataset with name {dset} exists!")
    else:
        raise Exception("Either --all_dsets or --dset option is required")

    inference_experiments = get_inference_experiments()
    training_experiments = get_training_experiments()
    scheduled_inf_exp = {}
    scheduled_train_exp = {}

    if all_exps:
        scheduled_train_exp = training_experiments
        scheduled_inf_exp = inference_experiments
    elif all_inference_exps:
        scheduled_inf_exp = inference_experiments
    elif exp is not None:
        train_exp = training_experiments.get(exp)
        inf_exp = inference_experiments.get(exp)
        if train_exp is not None:
            scheduled_train_exp[exp] = train_exp
        elif inf_exp is not None:
            scheduled_inf_exp[exp] = inf_exp
        else:
            raise Exception(f"No experiment with name {exp} exists!")
    else:
        raise Exception("Either --all_exps, --all_inference_exps or --exp option is required")

    for dset_name_dict, (dset_loader, dset_name) in scheduled_dataset_loaders.items():
        X, y = dset_loader()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
        train_exp_results = []
        for exp_name_dict, (exp_func, exp_name) in scheduled_train_exp.items():
            if redraw:
                redraw_decision_boundary(exp_name, dset_name, X_test, y_test)
            else:
                results = []
                for i in range(execs):
                    print(
                        f"Running '{exp_name}' training experiment on '{dset_name}' dataset [{i + 1} of {execs}]..."
                    )
                    result, plot_data = exp_func(X_train, X_test, y_train, y_test)
                    shutil.rmtree("/tmp/fhe_keys_client", True)
                    results.append(result)
                final_result = evaluate_experiment_results(
                    results, dset_name, dset_name_dict, exp_name, exp_name_dict
                )
                if draw:
                    draw_decision_boundary(plot_data, exp_name, dset_name, X_test, y_test)
                print(
                    f"Mean result of {execs} executions of '{exp_name}' training experiment on '{dset_name}' dataset:"
                )
                print(final_result)
                print(
                    f"Training on encrypted data with FHE was {final_result.fhe_duration_processing / final_result.clear_duration} times slower than normal inference on clear data"
                )
                train_exp_results.append(final_result)
        if len(train_exp_results) > 0:
            write_result_to_csv("training_experiments", train_exp_results)
            if draw or redraw:
                draw_feature_dim_runtime_plot("training_experiments", "synth_")
                draw_feature_dim_runtime_plot("training_experiments", "spam_")

        inf_exp_results = []
        for exp_name_dict, (exp_func, exp_name) in scheduled_inf_exp.items():
            if redraw:
                redraw_decision_boundary(exp_name, dset_name, X_test, y_test)
            else:
                results = []
                for i in range(execs):
                    print(
                        f"Running '{exp_name}' inference experiment on '{dset_name}' dataset [{i + 1} of {execs}]..."
                    )
                    result, plot_data = exp_func(X_train, X_test, y_train, y_test)
                    shutil.rmtree("/tmp/fhe_keys_client", True)
                    results.append(result)
                final_result = evaluate_experiment_results(
                    results, dset_name, dset_name_dict, exp_name, exp_name_dict
                )
                if draw:
                    draw_decision_boundary(plot_data, exp_name, dset_name, X_test, y_test)
                print(
                    f"Mean result of {execs} executions of '{exp_name}' inference experiment on '{dset_name}' dataset:"
                )
                print(final_result)
                print(
                    f"Inference on encrypted data with FHE was {final_result.fhe_duration_processing / final_result.clear_duration} times slower than normal inference on clear data"
                )
                inf_exp_results.append(final_result)
        if len(inf_exp_results) > 0:
            write_result_to_csv("inference_experiments", inf_exp_results)
            if draw or redraw:
                draw_feature_dim_runtime_plot("inference_experiments", "synth_")
                draw_feature_dim_runtime_plot("inference_experiments", "spam_")


if __name__ == "__main__":
    main()
