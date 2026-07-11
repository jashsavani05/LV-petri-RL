#!/usr/bin/env python3

import numpy as np
from numbers import Number
from scipy.stats import Normal
from matplotlib import pyplot as plt
from sklearn.linear_model import ridge_regression, LinearRegression
from lv_sim import LV_SIM
import pysindy


class STLSQ:
    def __init__(
        self,
        spn_model: LV_SIM,
        spn_model_expressions: {
            "hunt_rate": 2,
            "predator_death_rate": 1,
            "prey_birth_rate": 1,
        },
        library_functions=None,
        threshold=0.1,
        initial_guess=None,
        sparse_ind=None,
    ):
        self.spn_model = spn_model
        self.spn_model_targets = spn_model_expressions
        self.library_functions = library_functions
        self.threshold = threshold
        self.expressions = initial_guess

    def predict_(self, x0, spn_model, expressions, plot=False):
        pred_spn_model = spn_model(x0=x0, expression_functions=expressions)
        sim, y_labels, t = pred_spn_model.run_sim(plot=plot)
        Y_pred = np.stack(list(sim[y_label] for y_label in y_labels), axis=-1)
        return Y_pred, t

    def predict(self, x0, plot=False):
        return self.predict_(
            x0, spn_model=self.spn_model, expressions=self.expressions, plot=plot
        )

    def fit(self, x0, Y, t, weights=None):
        return self.coefficients
