import click
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from .experiment_collector import get_inference_experiments, get_training_experiments

@click.command()
@click.option("--run_all", is_flag=True, help="Run all training and inference experiments")
@click.option("--run_all_inference", is_flag=True, help="Run only the inference experiments")
@click.option("--run", type=str, required=False, help="Run only the specified experiment")
def main(run_all: bool, run_all_inference: bool, run: str | None):
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

    for exp_name,exp in scheduled_train_exp.items():
        print(f"Running {exp_name} training experiment...")
        result = exp(X_train, y_train)
        print(f"Results of {exp_name} training experiment:")
        print(result)
        print(f"Training on encrypted data with FHE was {result.duration_in_sec_fhe / result.duration_in_sec_clear} times slower than normal training on clear data")
    for exp_name,exp in scheduled_inf_exp.items():
        print(f"Running {exp_name} inference experiment...")
        result = exp(X_train, X_test, y_train, y_test)
        print(f"Results of {exp_name} inference experiment:")
        print(result)
        print(f"Inference on encrypted data with FHE was {result.duration_in_sec_fhe / result.duration_in_sec_clear} times slower than normal inference on clear data")

if __name__ == "__main__":
    main()
