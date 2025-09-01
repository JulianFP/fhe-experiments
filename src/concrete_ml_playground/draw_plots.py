import os
import pickle
import numpy as np

from concrete.ml.common.utils import FheMode
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from .dataset_collector import get_dataset_loaders
from .interfaces import DecisionBoundaryPlotData
from .csv_handler import read_csv


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
        with open(clear_pickle_path, "rb") as file:
            Z_clear = pickle.load(file)
        with open(fhe_pickle_path, "rb") as file:
            Z_fhe = pickle.load(file)

        plt.figure(figsize=(10, 7.5))
        plt.title(clear_title)
        plt.contourf(xx, yy, Z_clear, alpha=0.8, cmap="bwr")
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap="bwr", edgecolor="k")
        png_path = f"results/{clear_title}.png"
        plt.savefig(png_path)
        print(f"Saved decision boundary to {png_path}")

        plt.figure(figsize=(10, 7.5))
        plt.title(fhe_title)
        plt.contourf(xx, yy, Z_fhe, alpha=0.8, cmap="bwr")
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap="bwr", edgecolor="k")
        png_path = f"results/{fhe_title}.png"
        plt.savefig(png_path)
        print(f"Saved decision boundary to {png_path}")


def apply_pca(X):
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    x1_min, x1_max = X_reduced[:, 0].min() - 0.5, X_reduced[:, 0].max() + 0.5
    x2_min, x2_max = X_reduced[:, 1].min() - 0.5, X_reduced[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x1_min, x1_max, 500), np.linspace(x2_min, x2_max, 500))
    return pca, X_reduced, xx, yy


def draw_decision_boundary(
    plot_data: DecisionBoundaryPlotData, exp_name: str, dset_name: str, X, y
):
    pca, X_reduced, xx, yy = apply_pca(X)

    clear_title = __get_clear_title(exp_name, dset_name)
    fhe_title = __get_fhe_title(exp_name, dset_name)
    os.makedirs("results", exist_ok=True)
    clear_pickle_path = f"results/{clear_title}.pickle"
    fhe_pickle_path = f"results/{fhe_title}.pickle"

    Z_clear = plot_data.clear_model.predict(
        pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
    ).reshape(xx.shape)
    with open(clear_pickle_path, "wb") as file:
        pickle.dump(Z_clear, file)

    if plot_data.fhe_trained_model is not None:
        Z_fhe = plot_data.fhe_trained_model.predict(
            pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
        ).reshape(xx.shape)
    elif plot_data.fhe_model is not None:
        Z_fhe = plot_data.fhe_model.predict(
            pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]), fhe=FheMode.SIMULATE
        ).reshape(xx.shape)
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
    _, X_reduced, xx, yy = apply_pca(X)
    draw_decision_boundary_from_pickle_files(
        exp_name, dset_name, X_reduced, y, xx, yy, clear_pickle_path, fhe_pickle_path
    )


def draw_feature_dim_runtime_plot(csv_file: str, dset_prefix: str):
    dataset_loaders = get_dataset_loaders()
    results = read_csv(csv_file)
    x = []
    y_clear = []
    y_clear_stdev = []
    y_pre = []
    y_pre_stdev = []
    y_fhe = []
    y_fhe_stdev = []
    y_post = []
    y_post_stdev = []
    experiments = [d.exp_name for d in results]
    for exp_name in experiments:
        for result in results:
            if result.exp_name == exp_name and result.dset_name_dict.startswith(dset_prefix):
                X, _ = dataset_loaders[result.dset_name_dict][0]()
                x.append(len(X[0]))
                y_clear.append(result.clear_duration)
                y_clear_stdev.append(result.clear_duration_stdev)
                y_pre.append(result.fhe_duration_preprocessing)
                y_pre_stdev.append(result.fhe_duration_preprocessing_stdev)
                y_fhe.append(result.fhe_duration_processing)
                y_fhe_stdev.append(result.fhe_duration_processing_stdev)
                y_post.append(result.fhe_duration_postprocessing)
                y_post_stdev.append(result.fhe_duration_postprocessing_stdev)

        plt.figure(figsize=(10, 7.5))
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
        print(f"Saved feature-runtime plot to {png_path}")
