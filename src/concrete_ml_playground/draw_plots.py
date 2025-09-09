import os
import pickle
import math
import numpy as np

from concrete.ml.common.utils import FheMode
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from . import logger
from .dataset_collector import get_dataset_loaders
from .interfaces import DecisionBoundaryPlotData
from .csv_handler import read_csv

figsize = (10, 7.5)


def __get_clear_title(exp_name: str, dset_name: str):
    return f"{exp_name} - {dset_name} - clear"


def __get_fhe_title(exp_name: str, dset_name: str):
    return f"{exp_name} - {dset_name} - FHE"


def draw_decision_boundary_from_pickle_files(
    exp_name: str, dset_name: str, X_reduced, y, xx, yy, clear_pickle_path, fhe_pickle_path
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

        plt.figure(figsize=figsize)
        plt.title(clear_title)
        plt.contourf(xx, yy, Z_clear, alpha=0.8, cmap="bwr")
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap="bwr", edgecolor="k")
        png_path = f"results/{clear_title}.png"
        plt.savefig(png_path)
        logger.info(f"Saved decision boundary to {png_path}")

        plt.figure(figsize=figsize)
        plt.title(fhe_title)
        plt.contourf(xx, yy, Z_fhe, alpha=0.8, cmap="bwr")
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap="bwr", edgecolor="k")
        png_path = f"results/{fhe_title}.png"
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
        np.linspace(x1_min - x1_padding, x1_max + x1_padding, width_samples),
        np.linspace(
            x2_min - x2_padding,
            x2_max + x2_padding,
            math.ceil(width_samples / figsize_aspect_ratio),
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
    plot_data: DecisionBoundaryPlotData, exp_name: str, dset_name: str, X, y
):
    logger.info(
        f"Running required computations for drawing the decision boundary for experiment '{exp_name}' on dataset '{dset_name}'..."
    )
    pca, X_reduced, xx, yy = apply_pca_if_necessary(X)

    clear_title = __get_clear_title(exp_name, dset_name)
    fhe_title = __get_fhe_title(exp_name, dset_name)
    clear_pickle_path = f"results/{clear_title}.pickle"
    fhe_pickle_path = f"results/{fhe_title}.pickle"

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
        exp_name, dset_name, X_reduced, y, xx, yy, clear_pickle_path, fhe_pickle_path
    )


def redraw_decision_boundary(exp_name: str, dset_name: str, X, y):
    clear_title = __get_clear_title(exp_name, dset_name)
    fhe_title = __get_fhe_title(exp_name, dset_name)
    clear_pickle_path = f"results/{clear_title}.pickle"
    fhe_pickle_path = f"results/{fhe_title}.pickle"
    _, X_reduced, xx, yy = apply_pca_if_necessary(X)
    draw_decision_boundary_from_pickle_files(
        exp_name, dset_name, X_reduced, y, xx, yy, clear_pickle_path, fhe_pickle_path
    )


def draw_feature_dim_runtime_plot(dset_prefix: str):
    dataset_loaders = get_dataset_loaders()
    results = read_csv()
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

        png_path = f"results/feature-runtime-plot_{exp_name}_{dset_prefix}.png"
        plt.savefig(png_path)
        logger.info(f"Saved feature-runtime plot to {png_path}")
