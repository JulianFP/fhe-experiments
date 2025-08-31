import csv

from dataclasses import asdict

from .interfaces import ExperimentResult


def write_result_to_csv(file_name: str, exp_results: dict[str, ExperimentResult]):
    with open(f"results/{file_name}.csv", mode="w") as csv_file:
        first_result = asdict(list(exp_results.values())[0])
        result_writer = csv.DictWriter(
            csv_file, fieldnames=["experiment"] + list(first_result.keys())
        )
        result_writer.writeheader()

        for exp_name, exp_result in exp_results.items():
            result_dict = asdict(exp_result)
            result_writer.writerow({"experiment": exp_name, **result_dict})
