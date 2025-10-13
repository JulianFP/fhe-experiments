import statistics
import numpy.typing as npt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from .interfaces import ExperimentOutput, ExperimentResult, ExperimentResultFinal

field_names = ExperimentResult.model_fields.keys()


def experiment_output_processor(y_true: npt.NDArray, exp_out: ExperimentOutput) -> ExperimentResult:
    label_count = np.unique(y_true)
    if label_count.size > 2:
        average_type = "micro"
    else:
        average_type = "binary"
    return ExperimentResult(
        accuracy_fhe=accuracy_score(y_true, exp_out.y_pred_fhe),
        accuracy_clear=accuracy_score(y_true, exp_out.y_pred_clear),
        f1_score_fhe=f1_score(y_true, exp_out.y_pred_fhe, average=average_type),
        f1_score_clear=f1_score(y_true, exp_out.y_pred_clear, average=average_type),
        clear_duration=exp_out.timings[1] - exp_out.timings[0],
        fhe_duration_preprocessing=exp_out.timings[3] - exp_out.timings[2],
        fhe_duration_processing=exp_out.timings[5] - exp_out.timings[4],
        fhe_duration_postprocessing=exp_out.timings[7] - exp_out.timings[6],
    )


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
            final_dict[key] = statistics.fmean(values)
            final_dict[f"{key}_stdev"] = statistics.stdev(values)

    return ExperimentResultFinal(
        dset_name=dset_name,
        dset_name_dict=dset_name_dict,
        exp_name=exp_name,
        exp_name_dict=exp_name_dict,
        **final_dict,
    )
