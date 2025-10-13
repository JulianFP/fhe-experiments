import os
import pickle
import math
import numpy as np

from concrete.ml.common.utils import FheMode
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from . import logger
from .dataset_collector import get_dataset_loaders
from .interfaces import ExperimentOutput
from .csv_handler import read_csv


plt.rcParams.update({"font.size": 16})
figsize = (10, 7.5)
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
        plt.title(clear_title)
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
        png_path = f"{results_dir}/{clear_title}.png"
        plt.savefig(png_path)
        logger.info(f"Saved decision boundary to {png_path}")

        plt.figure(figsize=figsize)
        plt.title(fhe_title)
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
        png_path = f"{results_dir}/{fhe_title}.png"
        plt.savefig(png_path)
        logger.info(f"Saved decision boundary to {png_path}")


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

    Z_clear = plot_data.clear_model.predict(inp).reshape(xx.shape)
    with open(clear_pickle_path, "wb") as file:
        pickle.dump(Z_clear, file)

    if plot_data.fhe_trained_model is not None:
        Z_fhe = plot_data.fhe_trained_model.predict(inp).reshape(xx.shape)
    elif plot_data.fhe_model is not None:
        Z_fhe = plot_data.fhe_model.predict(inp, fhe=FheMode.SIMULATE).reshape(xx.shape)
    else:
        raise Exception(
            "DecisionBoundaryPlotData needs to either have an fhe_trained_model or an fhe_model!"
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
    plt.title(title)
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
    png_path = f"{results_dir}/{title}.png"
    plt.savefig(png_path)
    logger.info(f"Saved dataset plot to {png_path}")


def draw_feature_dim_runtime_plot(results_dir: str, dset_prefix: str):
    dataset_loaders = get_dataset_loaders()
    results = read_csv(results_dir)
    experiments = set([d.exp_name for d in results])
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
        plt.title(f"Feature space dim - runtime: {exp_name}, {dset_prefix}")
        plt.xlabel("Dim of feature vectors")
        plt.ylabel("Runtime (in seconds)")
        plt.errorbar(x, y_clear, y_clear_stdev, fmt="bo-", label="clear")
        plt.errorbar(x, y_pre, y_pre_stdev, fmt="yo-", label="FHE pre")
        plt.errorbar(x, y_fhe, y_fhe_stdev, fmt="ro-", label="FHE")
        plt.errorbar(x, y_post, y_post_stdev, fmt="mo-", label="FHE post")
        plt.figlegend()

        png_path = f"{results_dir}/feature-runtime-plot_{exp_name}_{dset_prefix}.png"
        plt.savefig(png_path)
        logger.info(f"Saved feature-runtime plot to {png_path}")
