import os
import pickle
import numpy as np

from concrete.ml.common.utils import FheMode
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from .interfaces import DecisionBoundaryPlotData


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

        fig = plt.figure(figsize=(10, 7.5))
        plt.title(clear_title)
        plt.contourf(xx, yy, Z_clear, alpha=0.8, cmap="bwr")
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap="bwr", edgecolor="k")
        png_path = f"results/{clear_title}.png"
        fig.savefig(png_path)
        print(f"Saved decision boundary to {png_path}")

        fig = plt.figure(figsize=(10, 7.5))
        plt.title(fhe_title)
        plt.contourf(xx, yy, Z_fhe, alpha=0.8, cmap="bwr")
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap="bwr", edgecolor="k")
        png_path = f"results/{fhe_title}.png"
        fig.savefig(png_path)
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
