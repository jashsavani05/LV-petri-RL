#!/usr/bin/env python3

import numpy as np
from numbers import Number
from scipy.stats import Normal
from matplotlib import pyplot as plt
from sklearn.linear_model import ridge_regression, LinearRegression
from lv_sim import LV_SIM
from scipy.interpolate import interp1d
import pygad
import pysindy as ps
from itertools import islice
from tqdm import tqdm
from scipy.signal import savgol_filter


class ExpressionFunction:
    def __init__(
        self,
        num_args,
        library_functions=ps.PolynomialLibrary(degree=2),
        coefficients=None,
    ):
        self.num_args = num_args
        self.library_functions = library_functions
        self.coefficients = coefficients
        self.f = self.library_functions.fit(np.random.rand(1, self.num_args))
        self.num_coefficients = len(self.f.get_feature_names())
        if self.coefficients is None:
            self.coefficients = np.ones((self.num_coefficients, 1))

    def func(self, *args):
        try:
            self.f = self.library_functions.fit(np.random.rand(1, self.num_args))
            theta = self.f.transform(np.array([args]))
            return (theta @ self.coefficients)[0][0]
        except Exception as E:
            print(E)
            breakpoint()


class STLSQ:
    def __init__(
        self,
        spn_model=LV_SIM,
        spn_model_expressions={
            "hunt_rate": 2,
            "predator_death_rate": 1,
            "prey_birth_rate": 1,
        },
        library_functions=None,
        threshold=0.1,
        initial_guess=None,
        sparse_ind=None,
        interpolation="cubic",
    ):
        self.spn_model = spn_model
        self.spn_model_targets = spn_model_expressions
        self.library_functions = library_functions
        self.threshold = threshold
        self.expressions = initial_guess
        self.interpolation = interpolation
        if self.library_functions is None:
            self.library_functions = pysindy.PolynomialLibrary
        efs = {
            k: ExpressionFunction(num_args=v) for k, v in self.spn_model_targets.items()
        }

        self.num_coefficients = {k: ef.num_coefficients for k, ef in efs.items()}
        self.total_coefficients = sum(v for v in self.num_coefficients.values())


def run():
    x0 = (50, 150)

    lv_cpn = LV_SIM(x0=x0)

    sim, _, _ = lv_cpn.run_sim(max_steps=20000, seed=0, plot=True)

    indices = list(range(len(sim["time"])))[0::10]

    X = np.stack(
        (np.take(sim["Predator"], indices), np.take(sim["Prey"], indices)), axis=-1
    )

    feature_library = ps.feature_library.CustomLibrary(
        library_functions=[
            lambda x: x,
            lambda x: x * x,
            lambda y: y * y,
            lambda x, y: x * y,
        ]
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


# model = run()

ls = ps.feature_library.CustomLibrary(
    library_functions=[
        lambda x: x,
        lambda x: x * x,
        lambda x, y: x * y,
    ]
)

ls = ls.fit(np.array([1, 2]))
print(ls.get_feature_names())
ls.transform(np.array([[1, 2]])).sum()
