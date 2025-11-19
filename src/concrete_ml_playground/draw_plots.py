import os
import pickle
import math
import numpy as np

from concrete.ml.common.utils import FheMode
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from . import logger
from .dataset_collector import get_dataset_loaders
from .experiment_collector import get_inference_experiments, get_training_experiments
from .interfaces import ExperimentOutput, ExperimentResultFinal
from .csv_handler import read_csv
from .statistics_handler import RatioResult, calculate_runtime_ratios


plt.rcParams.update({"font.size": 16})
figsize = (10, 7.5)
barWidth = 0.25
groupSpacing = 0.25
label_cmap = "brg"


def __get_clear_title(exp_name: str, dset_name: str):
    return f"{exp_name} - {dset_name} - clear"


def __get_fhe_title(exp_name: str, dset_name: str):
    return f"{exp_name} - {dset_name} - FHE"


def draw_decision_boundary_from_pickle_files(
    results_dir: str,
    exp_name: str,
    dset_name: str,
    X_reduced,
    y,
    xx,
    yy,
    clear_pickle_path,
    fhe_pickle_path,
):
    clear_title = __get_clear_title(exp_name, dset_name)
    fhe_title = __get_fhe_title(exp_name, dset_name)

    if os.path.isfile(clear_pickle_path) and os.path.isfile(fhe_pickle_path):
        logger.info(
            "Loading computation results from disk and drawing the decision boundary for experiment '{exp_name}' on dataset '{dset_name}'..."
        )
        with open(clear_pickle_path, "rb") as file:
            Z_clear = pickle.load(file)
        with open(fhe_pickle_path, "rb") as file:
            Z_fhe = pickle.load(file)
        labels = np.unique(y)
        cmap = plt.cm.get_cmap(label_cmap, labels.size)

        plt.figure(figsize=figsize)
        # plt.title(clear_title)
        plt.contourf(xx, yy, Z_clear, alpha=0.5, cmap=cmap)
        for label in labels:
            plt.scatter(
                X_reduced[y == label, 0],
                X_reduced[y == label, 1],
                c=cmap(label),
                edgecolor="k",
                label=f"test set - label '{label}'",
            )
        plt.figlegend()
        plot_path = f"{results_dir}/{clear_title}.pdf"
        plt.savefig(plot_path, format="pdf")
        plt.close()
        logger.info(f"Saved decision boundary to {plot_path}")

        plt.figure(figsize=figsize)
        # plt.title(fhe_title)
        plt.contourf(xx, yy, Z_fhe, alpha=0.5, cmap=cmap)
        for label in labels:
            plt.scatter(
                X_reduced[y == label, 0],
                X_reduced[y == label, 1],
                c=cmap(label),
                edgecolor="k",
                label=f"test set - label '{label}'",
            )
        plt.figlegend()
        plot_path = f"{results_dir}/{fhe_title}.pdf"
        plt.savefig(plot_path, format="pdf")
        plt.close()
        logger.info(f"Saved decision boundary to {plot_path}")


def get_meshgrid(X_reduced):
    x1_min, x1_max = X_reduced[:, 0].min(), X_reduced[:, 0].max()
    x2_min, x2_max = X_reduced[:, 1].min(), X_reduced[:, 1].max()
    x1_padding = 0.05 * (x1_max - x1_min)
    x2_padding = 0.05 * (x2_max - x2_min)
    figsize_aspect_ratio = figsize[0] / figsize[1]
    width_samples = 500
    return np.meshgrid(
        np.linspace(x1_min - x1_padding, x1_max + x1_padding, width_samples, dtype=np.float32),
        np.linspace(
            x2_min - x2_padding,
            x2_max + x2_padding,
            math.ceil(width_samples / figsize_aspect_ratio),
            dtype=np.float32,
        ),
    )


def apply_pca_if_necessary(X):
    if len(X[0]) > 2:
        logger.info("Features have more than 2 dimensions. Applying PCA...")
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)
        xx, yy = get_meshgrid(X_reduced)
        return pca, X_reduced, xx, yy
    else:
        logger.info(f"Features have {len(X[0])} dimensions. Not applying PCA")
        xx, yy = get_meshgrid(X)
        return None, X, xx, yy


def draw_decision_boundary(
    results_dir: str, plot_data: ExperimentOutput, exp_name: str, dset_name: str, X, y
):
    logger.info(
        f"Running required computations for drawing the decision boundary for experiment '{exp_name}' on dataset '{dset_name}'..."
    )
    pca, X_reduced, xx, yy = apply_pca_if_necessary(X)

    clear_title = __get_clear_title(exp_name, dset_name)
    fhe_title = __get_fhe_title(exp_name, dset_name)
    clear_pickle_path = f"{results_dir}/{clear_title}.pickle"
    fhe_pickle_path = f"{results_dir}/{fhe_title}.pickle"

    if pca is not None:
        inp = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
    else:
        inp = np.c_[xx.ravel(), yy.ravel()]

    if plot_data.data_preparation_step is not None:
        inp = plot_data.data_preparation_step(inp)

    if plot_data.clear_model is not None:
        Z_clear = plot_data.clear_model.predict(inp).reshape(xx.shape)
    else:
        raise Exception(
            "ExperimentOutput needs to have a clear_model to plot a decision boundary out of it!"
        )
    with open(clear_pickle_path, "wb") as file:
        pickle.dump(Z_clear, file)

    if plot_data.fhe_trained_model is not None:
        Z_fhe = plot_data.fhe_trained_model.predict(inp).reshape(xx.shape)
    elif plot_data.fhe_model is not None:
        Z_fhe = plot_data.fhe_model.predict(inp, fhe=FheMode.SIMULATE).reshape(xx.shape)
    else:
        raise Exception(
            "ExperimentOutput needs to either have an fhe_trained_model or an fhe_model to plot a decision boundary out of it!"
        )
    with open(fhe_pickle_path, "wb") as file:
        pickle.dump(Z_fhe, file)

    draw_decision_boundary_from_pickle_files(
        results_dir, exp_name, dset_name, X_reduced, y, xx, yy, clear_pickle_path, fhe_pickle_path
    )


def redraw_decision_boundary(results_dir: str, exp_name: str, dset_name: str, X, y):
    clear_title = __get_clear_title(exp_name, dset_name)
    fhe_title = __get_fhe_title(exp_name, dset_name)
    clear_pickle_path = f"{results_dir}/{clear_title}.pickle"
    fhe_pickle_path = f"{results_dir}/{fhe_title}.pickle"
    _, X_reduced, xx, yy = apply_pca_if_necessary(X)
    draw_decision_boundary_from_pickle_files(
        results_dir, exp_name, dset_name, X_reduced, y, xx, yy, clear_pickle_path, fhe_pickle_path
    )


def draw_dataset(results_dir: str, dset_name: str, X_train, X_test, y_train, y_test):
    logger.info(f"Drawing dataset {dset_name}...")
    pca, X_reduced, _, _ = apply_pca_if_necessary(np.concatenate((X_train, X_test), axis=0))
    X_train_reduced, X_test_reduced = X_reduced[: len(X_train)], X_reduced[len(X_train) :]

    title = f"'{dset_name}' dataset"
    if pca is not None:
        title += " - with PCA applied"

    plt.figure(figsize=figsize)
    # plt.title(title)
    labels = np.unique(y_test)
    cmap = plt.cm.get_cmap(label_cmap, labels.size)
    for label in labels:
        plt.scatter(
            X_train_reduced[y_train == label, 0],
            X_train_reduced[y_train == label, 1],
            c=cmap(label),
            edgecolor="k",
            marker="^",
            label=f"train set - label '{label}'",
        )
        plt.scatter(
            X_test_reduced[y_test == label, 0],
            X_test_reduced[y_test == label, 1],
            c=cmap(label),
            edgecolor="k",
            marker="o",
            label=f"test set - label '{label}'",
        )
    plt.figlegend()

    plot_path = f"{results_dir}/{title}.pdf"
    plt.savefig(plot_path, format="pdf")
    plt.close()
    logger.info(f"Saved dataset plot to {plot_path}")


def draw_feature_dim_runtime_plot(results_dir: str, dset_prefix: str):
    dataset_loaders = get_dataset_loaders()
    experiment_loaders = {**get_inference_experiments(), **get_training_experiments()}
    results = read_csv(results_dir)
    experiments = set([d.exp_name for d in results if d.exp_name_dict in experiment_loaders])
    for exp_name in experiments:
        logger.info(
            f"Drawing feature_dim_runtime plot for experiment '{exp_name}' and datasets with prefix '{dset_prefix}'..."
        )
        x = []
        y_clear = []
        y_clear_stdev = []
        y_pre = []
        y_pre_stdev = []
        y_fhe = []
        y_fhe_stdev = []
        y_post = []
        y_post_stdev = []
        for result in results:
            if result.exp_name == exp_name and result.dset_name_dict.startswith(dset_prefix):
                X, _, _, _ = dataset_loaders[result.dset_name_dict][0]()
                x.append(len(X[0]))
                y_clear.append(result.clear_duration)
                y_clear_stdev.append(result.clear_duration_stdev)
                y_pre.append(result.fhe_duration_preprocessing)
                y_pre_stdev.append(result.fhe_duration_preprocessing_stdev)
                y_fhe.append(result.fhe_duration_processing)
                y_fhe_stdev.append(result.fhe_duration_processing_stdev)
                y_post.append(result.fhe_duration_postprocessing)
                y_post_stdev.append(result.fhe_duration_postprocessing_stdev)

        plt.figure(figsize=figsize)
        # plt.title(f"Feature space dim - runtime: {exp_name}, {dset_prefix}")
        plt.errorbar(
            x, y_clear, y_clear_stdev, fmt="o-", color="tab:green", label="clear", capsize=4
        )
        plt.errorbar(
            x, y_pre, y_pre_stdev, fmt="o-", color="tab:orange", label="FHE pre", capsize=4
        )
        plt.errorbar(x, y_fhe, y_fhe_stdev, fmt="o-", color="tab:red", label="FHE", capsize=4)
        plt.errorbar(
            x, y_post, y_post_stdev, fmt="o-", color="tab:pink", label="FHE post", capsize=4
        )
        plt.yscale("log")
        plt.xlabel("Dimensionality of feature vectors", fontweight="bold")
        plt.ylabel("Avg. Runtime (in seconds)", fontweight="bold")
        plt.tight_layout()
        plt.figlegend()

        plot_path = f"{results_dir}/feature-runtime-plot_{exp_name}_{dset_prefix}.pdf"
        plt.savefig(plot_path, format="pdf")
        plt.close()
        logger.info(f"Saved feature-runtime plot to {plot_path}")


def draw_runtime_plot(
    plot_path: str, results: list[ExperimentResultFinal], result_attr_name: str, xlabel: str
):
    result_names = []
    clear_dur = []
    clear_dur_stdev = []
    pre_dur = []
    pre_dur_stdev = []
    fhe_dur = []
    fhe_dur_stdev = []
    post_dur = []
    post_dur_stdev = []
    for r in results:
        result_names.append(getattr(r, result_attr_name))
        clear_dur.append(r.clear_duration)
        clear_dur_stdev.append(r.clear_duration_stdev)
        pre_dur.append(r.fhe_duration_preprocessing)
        pre_dur_stdev.append(r.fhe_duration_preprocessing_stdev)
        fhe_dur.append(r.fhe_duration_processing)
        fhe_dur_stdev.append(r.fhe_duration_processing_stdev)
        post_dur.append(r.fhe_duration_postprocessing)
        post_dur_stdev.append(r.fhe_duration_postprocessing_stdev)

    br1 = np.arange(len(clear_dur)) * (barWidth * 4 + groupSpacing)
    br2 = br1 + barWidth
    br3 = br2 + barWidth
    br4 = br3 + barWidth

    plt.figure(figsize=figsize)
    plt.bar(
        br1,
        clear_dur,
        color="tab:green",
        width=barWidth,
        label="clear",
        yerr=clear_dur_stdev,
        error_kw=dict(capsize=4, capthick=2, elinewidth=2),
        capsize=4,
    )
    plt.bar(
        br2,
        pre_dur,
        color="tab:orange",
        width=barWidth,
        label="FHE pre",
        yerr=pre_dur_stdev,
        error_kw=dict(capsize=4, capthick=2, elinewidth=2),
        capsize=4,
    )
    plt.bar(
        br3,
        fhe_dur,
        color="tab:red",
        width=barWidth,
        label="FHE proc",
        yerr=fhe_dur_stdev,
        error_kw=dict(capsize=4, capthick=2, elinewidth=2),
        capsize=4,
    )
    plt.bar(
        br4,
        post_dur,
        color="tab:pink",
        width=barWidth,
        label="FHE post",
        yerr=post_dur_stdev,
        error_kw=dict(capsize=4, capthick=2, elinewidth=2),
        capsize=4,
    )
    plt.yscale("log")
    plt.xlabel(xlabel, fontweight="bold")
    plt.ylabel("Avg. Runtime (in seconds)", fontweight="bold")
    if len(result_names) > 2:
        rotation = 45
    else:
        rotation = 0
    plt.xticks(br1 + 1.5 * barWidth, result_names, rotation=rotation)
    plt.tight_layout()
    plt.legend()

    plt.savefig(plot_path, format="pdf")
    plt.close()
    logger.info(f"Saved runtime plot to {plot_path}")


def draw_runtime_plot_with_ratios(
    plot_path: str, results: list[RatioResult], result_attr_name: str, xlabel: str
):
    logger.info(f"Drawing runtime plot with ratios with the following ratios: {results}")

    result_names = []
    fhe_proc = []
    fhe_proc_stdev = []
    fhe_pre_post = []
    fhe_pre_post_stdev = []
    for r in results:
        result_names.append(getattr(r, result_attr_name))
        fhe_proc.append(r.fhe_proc_to_clear_proc)
        fhe_proc_stdev.append(r.fhe_proc_to_clear_proc_stdev)
        fhe_pre_post.append(r.fhe_pre_and_post_to_clear_proc)
        fhe_pre_post_stdev.append(r.fhe_pre_and_post_to_clear_proc_stdev)

    br1 = np.arange(len(fhe_proc)) * (barWidth * 2 + groupSpacing)
    br2 = br1 + barWidth

    plt.figure(figsize=figsize)
    proc_bars = plt.bar(
        br1,
        fhe_proc,
        color="tab:red",
        width=barWidth,
        label="FHE proc",
        yerr=fhe_proc_stdev,
        error_kw=dict(capsize=4, capthick=2, elinewidth=2),
        capsize=4,
    )
    pre_post_bars = plt.bar(
        br2,
        fhe_pre_post,
        color="tab:orange",
        width=barWidth,
        label="FHE pre + FHE post",
        yerr=fhe_pre_post,
        error_kw=dict(capsize=4, capthick=2, elinewidth=2),
        capsize=4,
    )
    for bar, ratio in zip(proc_bars + pre_post_bars, fhe_proc + fhe_pre_post):
        height = bar.get_height()
        factor = f"{round(ratio):,}"
        plt.text(
            bar.get_x() + 3 / 4 * bar.get_width(),
            height,
            f"{factor}x slower",
            ha="left",
            va="bottom",
            rotation=45,
            fontsize=11,
            color="tab:blue",
        )
    plt.xlabel(xlabel, fontweight="bold")
    plt.ylabel("Avg. Runtime relative to clear runtime", fontweight="bold")
    if len(result_names) > 2:
        rotation = 45
    else:
        rotation = 0
    plt.xticks(br1 + 0.5 * barWidth, result_names, rotation=rotation)
    plt.tight_layout()
    plt.legend()

    plt.savefig(plot_path, format="pdf")
    plt.close()
    logger.info(f"Saved runtime plot with ratios to {plot_path}")


def draw_acc_f1_plot(
    plot_path: str, results: list[ExperimentResultFinal], result_attr_name: str, xlabel: str
):
    result_names = []
    clear_acc, clear_acc_stdev = [], []
    clear_f1, clear_f1_stdev = [], []
    fhe_acc, fhe_acc_stdev = [], []
    fhe_f1, fhe_f1_stdev = [], []

    for r in results:
        result_names.append(getattr(r, result_attr_name))
        clear_acc.append(r.accuracy_clear)
        clear_acc_stdev.append(r.accuracy_clear_stdev)
        clear_f1.append(r.f1_score_clear)
        clear_f1_stdev.append(r.f1_score_clear_stdev)
        fhe_acc.append(r.accuracy_fhe)
        fhe_acc_stdev.append(r.accuracy_fhe_stdev)
        fhe_f1.append(r.f1_score_fhe)
        fhe_f1_stdev.append(r.f1_score_fhe_stdev)

    subGroupSpacing = 0.25 * groupSpacing
    br1 = np.arange(len(clear_acc)) * (barWidth * 4 + groupSpacing + subGroupSpacing)
    br2 = br1 + barWidth
    br3 = br2 + barWidth + subGroupSpacing
    br4 = br3 + barWidth

    plt.figure(figsize=figsize)
    plt.bar(
        br1,
        clear_acc,
        color="tab:green",
        alpha=0.55,
        width=barWidth,
        label="Accuracy clear",
        yerr=clear_acc_stdev,
        error_kw=dict(capsize=4, capthick=2, elinewidth=2),
        capsize=4,
    )
    plt.bar(
        br2,
        fhe_acc,
        color="tab:red",
        alpha=0.55,
        width=barWidth,
        label="Accuracy FHE",
        yerr=fhe_acc_stdev,
        error_kw=dict(capsize=4, capthick=2, elinewidth=2),
        capsize=4,
    )
    plt.bar(
        br3,
        clear_f1,
        color="tab:green",
        width=barWidth,
        label="F1-Score clear",
        yerr=clear_f1_stdev,
        error_kw=dict(capsize=4, capthick=2, elinewidth=2),
        capsize=4,
    )
    plt.bar(
        br4,
        fhe_f1,
        color="tab:red",
        width=barWidth,
        label="F1-Score FHE",
        yerr=fhe_f1_stdev,
        error_kw=dict(capsize=4, capthick=2, elinewidth=2),
        capsize=4,
    )
    plt.xlabel(xlabel, fontweight="bold")
    if len(result_names) > 2:
        rotation = 45
    else:
        rotation = 0
    plt.xticks(br1 + 1.5 * barWidth + 0.5 * subGroupSpacing, result_names, rotation=rotation)
    plt.tight_layout()
    plt.legend()

    plt.savefig(plot_path, format="pdf")
    plt.close()
    logger.info(f"Saved accuracy/f1 plot to {plot_path}")


def draw_runtime_plots_per_exp_non_ner(results_dir: str):
    results = read_csv(results_dir)
    experiments = set([r.exp_name for r in results if "ner" not in r.exp_name_dict])
    for exp_name in experiments:
        logger.info(f"Drawing plots for experiment '{exp_name}'...")
        results_in_plot = []
        ratio_results_in_plot = []
        for r in results:
            filter_dset = False
            if r.dset_name_dict.startswith("synth_") or r.dset_name_dict.startswith("spam_"):
                prefix, dim = r.dset_name_dict.split("_")
                dim = int(dim)
                other_dims = [
                    int(re.dset_name_dict.split("_")[1])
                    for re in results
                    if re.exp_name == exp_name and re.dset_name_dict.startswith(prefix)
                ]
                if dim != min(other_dims) and dim != max(other_dims):
                    filter_dset = True
            if r.exp_name == exp_name and not filter_dset:
                results_in_plot.append(r)
                ratio_results_in_plot.append(calculate_runtime_ratios(r))
        draw_runtime_plot(
            f"{results_dir}/runtime-plot_{exp_name}.pdf",
            results_in_plot,
            "dset_name_dict",
            "Datasets",
        )
        draw_runtime_plot_with_ratios(
            f"{results_dir}/runtime-plot-with-ratio_{exp_name}.pdf",
            ratio_results_in_plot,
            "dset_name_dict",
            "Datasets",
        )
        draw_acc_f1_plot(
            f"{results_dir}/acc_f1-plot_{exp_name}.pdf",
            results_in_plot,
            "dset_name_dict",
            "Datasets",
        )


def draw_runtime_plot_ner(results_dir: str):
    results = read_csv(results_dir)
    results_in_plot = [
        r for r in results if "ner" in r.exp_name_dict and r.dset_name_dict == "cconll"
    ]
    ratio_results_in_plot = [calculate_runtime_ratios(r) for r in results_in_plot]
    if len(results_in_plot) > 0:
        logger.info("Drawing runtime plots for ner experiments...")
        draw_runtime_plot(
            f"{results_dir}/runtime-plot-ner.pdf",
            results_in_plot,
            "exp_name_dict",
            "NER Experiments on CCoNLL",
        )
        draw_runtime_plot_with_ratios(
            f"{results_dir}/runtime-plot-ner-with-ratio.pdf",
            ratio_results_in_plot,
            "exp_name_dict",
            "NER Experiments on CCoNLL",
        )
        draw_acc_f1_plot(
            f"{results_dir}/acc_f1-plot-ner.pdf",
            results_in_plot,
            "exp_name_dict",
            "NER Experiments on CCoNLL",
        )
