import click
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from concrete_ml_playground.interfaces import ExperimentResult

from .experiment_collector import get_inference_experiments, get_training_experiments


@click.command()
@click.option("--run_all", is_flag=True, help="Run all training and inference experiments")
@click.option("--run_all_inference", is_flag=True, help="Run only the inference experiments")
@click.option("--run", type=str, required=False, help="Run only the specified experiment")
@click.option(
    "--execs",
    type=int,
    default=1,
    show_default=True,
    help="How often each experiment should run. Will calculate the mean value between all executions",
)
def main(run_all: bool, run_all_inference: bool, run: str | None, execs: int):
    # Create the data for classification:
    X, y = make_classification(
        n_features=30,
        n_redundant=0,
        n_informative=2,
        random_state=2,
        n_clusters_per_class=1,
        n_samples=250,
    )
    # Retrieve train and test sets:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    inference_experiments = get_inference_experiments()
    training_experiments = get_training_experiments()
    scheduled_inf_exp = {}
    scheduled_train_exp = {}

    if run_all:
        scheduled_train_exp = training_experiments
        scheduled_inf_exp = inference_experiments
    elif run_all_inference:
        scheduled_inf_exp = inference_experiments
    elif run is not None:
        train_exp = training_experiments.get(run)
        inf_exp = inference_experiments.get(run)
        if train_exp is not None:
            scheduled_train_exp[run] = train_exp
        elif inf_exp is not None:
            scheduled_inf_exp[run] = inf_exp
        else:
            raise Exception(f"No experiment with name {run} exists!")
    else:
        raise Exception("Either --all, --all_inference or --run option is required")

    for exp_name, exp in scheduled_train_exp.items():
        mean_result = ExperimentResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)
        for i in range(execs):
            print(f"Running {exp_name} training experiment, {i+1}th execution...")
            mean_result += exp(X_train, X_test, y_train, y_test)
        mean_result = mean_result / execs
        print(f"Mean result of {execs} executions of {exp_name} training experiment:")
        print(mean_result)
        print(
            f"Training on encrypted data with FHE was {mean_result.fhe_duration_processing / mean_result.clear_duration} times slower than normal inference on clear data"
        )
    for exp_name, exp in scheduled_inf_exp.items():
        mean_result = ExperimentResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)
        for i in range(execs):
            print(f"Running {exp_name} inference experiment, {i+1}th execution...")
            mean_result += exp(X_train, X_test, y_train, y_test)
        mean_result = mean_result / execs
        print(f"Mean result of {execs} executions of {exp_name} inference experiment:")
        print(mean_result)
        print(
            f"Inference on encrypted data with FHE was {mean_result.fhe_duration_processing / mean_result.clear_duration} times slower than normal inference on clear data"
        )


if __name__ == "__main__":
    main()
