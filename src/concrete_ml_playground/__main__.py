import click
from sklearn.model_selection import train_test_split

from concrete_ml_playground.interfaces import ExperimentResult

from .dataset_collector import get_dataset_loaders
from .experiment_collector import get_inference_experiments, get_training_experiments


@click.command()
@click.option("--all_exps", is_flag=True, help="Run all training and inference experiments")
@click.option("--all_inference_exps", is_flag=True, help="Run only the inference experiments")
@click.option("--exp", type=str, required=False, help="Run only the specified experiment")
@click.option("--all_dsets", is_flag=True, help="Run on all datasets")
@click.option("--dset", type=str, required=False, help="Run only on the specified dataset")
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

    for dset_name, dset_loader in scheduled_dataset_loaders.items():
        X, y = dset_loader()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
        for exp_name, exp_func in scheduled_train_exp.items():
            mean_result = ExperimentResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)
            for i in range(execs):
                print(
                    f"Running {exp_name} training experiment on {dset_name} dataset, {i+1}th execution..."
                )
                mean_result += exp_func(X_train, X_test, y_train, y_test)
            mean_result = mean_result / execs
            print(f"Mean result of {execs} executions of {exp_name} training experiment:")
            print(mean_result)
            print(
                f"Training on encrypted data with FHE was {mean_result.fhe_duration_processing / mean_result.clear_duration} times slower than normal inference on clear data"
            )
        for exp_name, exp_func in scheduled_inf_exp.items():
            mean_result = ExperimentResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)
            for i in range(execs):
                print(
                    f"Running {exp_name} inference experiment on {dset_name} dataset, {i+1}th execution..."
                )
                mean_result += exp_func(X_train, X_test, y_train, y_test)
            mean_result = mean_result / execs
            print(f"Mean result of {execs} executions of {exp_name} inference experiment:")
            print(mean_result)
            print(
                f"Inference on encrypted data with FHE was {mean_result.fhe_duration_processing / mean_result.clear_duration} times slower than normal inference on clear data"
            )


if __name__ == "__main__":
    main()
