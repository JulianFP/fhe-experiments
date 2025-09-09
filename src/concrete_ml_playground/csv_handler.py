import csv
from . import logger
from .interfaces import ExperimentResultFinal

# this ensures that dset_name and exp_name are at the beginning
field_names = ["dset_name", "dset_name_dict", "exp_name", "exp_name_dict"]
field_names_dict = list(ExperimentResultFinal.model_fields.keys())
for sorted_name in field_names:
    field_names_dict.remove(sorted_name)
field_names_dict.sort()
arranged_field_names = field_names + field_names_dict


def get_file_name(results_dir: str) -> str:
    return f"{results_dir}/experiment_data.csv"


def init_csv(results_dir: str):
    with open(get_file_name(results_dir), "w", newline="") as csv_file:
        result_writer = csv.DictWriter(
            csv_file,
            fieldnames=arranged_field_names,
        )
        result_writer.writeheader()
    logger.info("Successfully initiated experiment_data.csv file")


def append_result_to_csv(results_dir: str, exp_result: ExperimentResultFinal):
    with open(get_file_name(results_dir), "a", newline="") as csv_file:
        result_writer = csv.DictWriter(
            csv_file,
            fieldnames=arranged_field_names,
        )
        result_writer.writerow(exp_result.model_dump())
    logger.info("Successfully appended experiment result to csv")


def read_csv(results_dir: str) -> list[ExperimentResultFinal]:
    with open(get_file_name(results_dir), "r", newline="") as csv_file:
        result_reader = csv.DictReader(
            csv_file,
        )
        results = []
        for row in result_reader:
            results.append(ExperimentResultFinal.model_validate(row))
        return results
