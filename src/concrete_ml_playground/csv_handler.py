import csv
import os

from .interfaces import ExperimentResultFinal

file_name = "results/experiment_data.csv"

# this ensures that dset_name and exp_name are at the beginning
field_names = ["dset_name", "dset_name_dict", "exp_name", "exp_name_dict"]
field_names_dict = list(ExperimentResultFinal.model_fields.keys())
for sorted_name in field_names:
    field_names_dict.remove(sorted_name)
field_names_dict.sort()
arranged_field_names = field_names + field_names_dict


def init_csv():
    if os.path.exists("results"):
        raise Exception(
            "The 'results' directory already exists. Please move it away to run a new experiment!"
        )

    os.makedirs("results")
    with open(file_name, "w", newline="") as csv_file:
        result_writer = csv.DictWriter(
            csv_file,
            fieldnames=arranged_field_names,
        )
        result_writer.writeheader()
    print("Successfully initiated result csv file")


def append_result_to_csv(exp_result: ExperimentResultFinal):
    with open(file_name, "a", newline="") as csv_file:
        result_writer = csv.DictWriter(
            csv_file,
            fieldnames=arranged_field_names,
        )
        result_writer.writerow(exp_result.model_dump())
    print("Successfully appended experiment result to csv")


def read_csv() -> list[ExperimentResultFinal]:
    with open(file_name, "r", newline="") as csv_file:
        result_reader = csv.DictReader(
            csv_file,
        )
        results = []
        for row in result_reader:
            results.append(ExperimentResultFinal.model_validate(row))
        return results
