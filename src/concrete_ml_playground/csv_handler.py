import csv

from .interfaces import ExperimentResultFinal


def write_result_to_csv(file_name: str, exp_results: list[ExperimentResultFinal]):
    # this ensures that dset_name and exp_name are at the beginning
    field_names = ["dset_name", "dset_name_dict", "exp_name", "exp_name_dict"]
    field_names_dict = list(ExperimentResultFinal.model_fields.keys())
    for sorted_name in field_names:
        field_names_dict.remove(sorted_name)
    field_names_dict.sort()
    arranged_field_names = field_names + field_names_dict

    with open(f"results/{file_name}.csv", "w", newline="") as csv_file:
        result_writer = csv.DictWriter(
            csv_file,
            fieldnames=arranged_field_names,
        )
        result_writer.writeheader()

        for result in exp_results:
            result_writer.writerow(result.model_dump())


def read_csv(file_name: str) -> list[ExperimentResultFinal]:
    with open(f"results/{file_name}.csv", "r", newline="") as csv_file:
        result_reader = csv.DictReader(
            csv_file,
        )
        results = []
        for row in result_reader:
            results.append(ExperimentResultFinal.model_validate(row))
        return results
