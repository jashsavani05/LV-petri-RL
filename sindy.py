#!/usr/bin/env python3

# %% [markdown]
# # Generating data and fitting pysindy models.
# [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dynamicslab/pysindy/v2.0?filepath=examples/tutorial_1.ipynb)
#
# This notebook discusses the relationship between SINDy and PySINDy,
# using a brief example showing how different objects in the SINDy method are represented in PySINDy.
# It may help to be familiar with the basics of SINDy, as presented in the original paper
# or [summarized here](https://pysindy.readthedocs.org/en/latest/summary.html).
#
# Suppose we have measurements of the position of a particle obeying the following dynamical system at different points in time:
#
# $$
#     \frac{d}{dt} \begin{bmatrix} x \\ y \end{bmatrix}
#     = \begin{bmatrix} -2x \\ y \end{bmatrix}
#     = \begin{bmatrix} -2 & 0 \\ 0 & 1 \end{bmatrix}
#     \begin{bmatrix} x \\ y \end{bmatrix}
# $$
#
# Note that this system of differential equations decouples into two differential equations whose solutions are simply $x(t) = x_0e^{-2t}$ and $y(t) = y_0e^t$, where $x_0 = x(0)$ and $y_0=y(0)$ are the initial conditions.
#
# Using the initial conditions $x_0 = 3$ and $y_0 = \tfrac{1}{2}$, we construct the data matrix $X$.
# %%
import numpy as np
from scipy.stats import Normal
import pysindy as ps
from numbers import Number
import matplotlib.pyplot as plt
import optimizers
from sklearn import linear_model
from lv_sim import setup_lotka_volterra_simulation, run_sim
from scipy.signal import savgol_filter
from importlib import reload


def run():

    lv_cpn = setup_lotka_volterra_simulation()

    sim = run_sim(lv_cpn, max_steps=20000, seed=0, plot=True)

    indices = list(range(len(sim["time"])))[0::10]

    X = np.stack(
        (np.take(sim["Predator"], indices), np.take(sim["Prey"], indices)), axis=-1
    )

    # X = np.stack(
    #     (
    #         savgol_filter(np.take(sim["Predator"], indices), 5, 2),
    #         savgol_filter(np.take(sim["Prey"], indices), 5, 2),
    #     ),
    #     axis=-1,
    # )
    # First column is x, second is y
    t = np.array(np.take(sim["time"], indices))
    print(f"Data is shape: {X.shape}")
    print(f"time is shape: {t.shape}")

    differentiation_method = ps.FiniteDifference(order=2)
    feature_library = ps.feature_library.CustomLibrary(
        library_functions=[lambda x: x, lambda x, y: x * y]
    )
    # optimizer = linear_model.Lasso(alpha=2, max_iter=10000, fit_intercept=False)
    optimizer = ps.optimizers.STLSQ(threshold=0.0001)
    model = ps.SINDy(
        differentiation_method=differentiation_method,
        feature_library=feature_library,
        optimizer=optimizer,
    )
    model.fit(X, t=t, feature_names=["x", "y"])
    model.print()
    indices_test = list(range(len(sim["time"])))[0::3]
    t_test = np.array(np.take(sim["time"], indices_test))
    x_preds = model.simulate([sim["Predator"][0], sim["Prey"][0]], t_test)
    x_act = model.differentiation_method(X, t)
    plt.plot(sim["time"], sim["Predator"], "blue")
    plt.plot(sim["time"], sim["Prey"], "red")
    plt.plot(t_test, x_preds[:, 0], "b--", label="model prediction")
    plt.plot(t_test, x_preds[:, 1], "r--", label="model prediction")
    plt.show()
    return model


model = run()
