# Experiments for by Bachelor Thesis

Some first results from running the command `poetry run python -u -m concrete_ml_playground --run_all --execs 10 | tee output`. clear_size and fhe_size not very reliable yet.

```
Mean result of 10 executions of sgd_classifier_training training experiment:
ExperimentResult(accuracy_fhe=0.7800000000000001, accuracy_clear=0.7800000000000001, fhe_duration_preprocessing=1.4751914739608765, fhe_duration_processing=568.6967245101929, fhe_duration_postprocessing=0.0224437952041626, clear_duration=0.02183201313018799, clear_size=2864, fhe_size=496)
Training on encrypted data with FHE was 26048.753320133972 times slower than normal inference on clear data
Mean result of 10 executions of logistical_regression inference experiment:
ExperimentResult(accuracy_fhe=0.7699999999999999, accuracy_clear=0.7699999999999999, fhe_duration_preprocessing=1.2221187114715577, fhe_duration_processing=0.26373326778411865, fhe_duration_postprocessing=0.06837193965911866, clear_duration=0.003414297103881836, clear_size=920, fhe_size=920)
Inference on encrypted data with FHE was 77.2437956510202 times slower than normal inference on clear data
```
