import statistics
from .interfaces import ExperimentResult, ExperimentResultFinal

field_names = ExperimentResult.model_fields.keys()


def evaluate_experiment_results(
    results: list[ExperimentResult],
    dset_name: str,
    dset_name_dict: str,
    exp_name: str,
    exp_name_dict: str,
) -> ExperimentResultFinal:
    result_dicts = []
    for result in results:
        result_dicts.append(result.model_dump())

    if len(results) < 2:
        final_dict = result_dicts[0]
        for key in field_names:
            final_dict[f"{key}_stdev"] = 0.0
    else:
        final_dict = {}
        for key in field_names:
            values = [d[key] for d in result_dicts]
            print(values)
            final_dict[key] = statistics.fmean(values)
            final_dict[f"{key}_stdev"] = statistics.stdev(values)

    return ExperimentResultFinal(
        dset_name=dset_name,
        dset_name_dict=dset_name_dict,
        exp_name=exp_name,
        exp_name_dict=exp_name_dict,
        **final_dict,
    )
