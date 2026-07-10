#!/usr/bin/env python3
import numpy as np
from numbers import Number
from scipy.stats import Normal
from matplotlib import pyplot as plt
from sklearn.linear_model import ridge_regression, LinearRegression
from sklearn.preprocessing import StandardScaler
import pysindy
from sklearn.utils.validation import check_is_fitted
import warnings
from typing import List, Tuple, Union, Optional
from abc import ABC


class GaussianDistribution(Normal, float):

    def norm(self) -> float:
        return (self.mu**2 + self.sigma**2) ** 0.5

    def __float__(self) -> "GaussianDistribution":
        return self

    def __init__(self, *args, **kwargs):
        super(Normal, self).__init__(*args, **kwargs)

    def do_plot(self, ax=None):
        ax = self.plot() if ax is None else ax
        sample = self.sample(10000)
        ax.hist(sample, density=True, bins=50, alpha=0.5)
        return ax

    def do_plot_and_show(self, ax=None):
        ax = self.do_plot(ax=ax)
        plt.show()
        return ax

    def __add__(self, something) -> "GaussianDistribution":
        if isinstance(something, Normal):
            return GaussianDistribution(
                mu=(self.mu + something.mu),
                sigma=((self.sigma**2 + something.sigma**2) ** 0.5),
            )
        elif isinstance(something, Number):
            return GaussianDistribution(mu=(self.mu + something), sigma=self.sigma)

    def __sub__(self, something) -> "GaussianDistribution":
        if isinstance(something, Normal):
            return GaussianDistribution(
                mu=(self.mu - something.mu),
                sigma=((self.sigma**2 + something.sigma**2) ** 0.5),
            )
        elif isinstance(something, Number):
            return GaussianDistribution(mu=(self.mu - something), sigma=self.sigma)

    def __mul__(self, something) -> "GaussianDistribution":
        if isinstance(something, Number):
            return GaussianDistribution(
                mu=(self.mu * something),
                sigma=((self.sigma**2) * (something**2)) ** 0.5,
            )

    def __truediv__(self, something) -> "GaussianDistribution":
        if isinstance(something, Number):
            something = 1 / something
            return GaussianDistribution(
                mu=(self.mu * something),
                sigma=((self.sigma**2) * (something**2)) ** 0.5,
            )


def gaussian_ridge_regression(
    X: np.ndarray, y: np.ndarray, alpha: float
) -> List[GaussianDistribution]:
    """
    Perform ridge regression with uncertainty propagation.
    Returns a list of scipy.stats.Normal objects for each coefficient.
    Assumes: y ~ N(Xw, sigma^2 I), w ~ N(0, (1/alpha) I) prior.
    Posterior is Gaussian: w | y ~ N( (X^T X + alpha I)^{-1} X^T y, (X^T X + alpha I)^{-1} sigma^2 )
    But we assume sigma^2 is unknown. So we estimate it from residuals.
    """
    n_samples, n_features = X.shape

    # Compute posterior mean and covariance
    XtX = X.T @ X
    XtX_reg = XtX + alpha * np.eye(n_features)

    try:
        inv_XtX_reg = np.linalg.inv(XtX_reg)
    except np.linalg.LinAlgError:
        # If singular, use pseudo-inverse
        inv_XtX_reg = np.linalg.pinv(XtX_reg)

    # Posterior mean
    mu_w = inv_XtX_reg @ X.T @ y

    # Estimate residual variance from data
    # Use the fact that residual sum of squares ~ chi^2(n_samples - rank)
    y_pred = X @ mu_w
    residuals = y - y_pred
    ss_res = np.sum(residuals**2)
    dof = n_samples - np.linalg.matrix_rank(XtX_reg)  # conservative estimate
    if dof <= 0:
        dof = 1
    sigma2_res = ss_res / dof

    # Posterior covariance matrix
    cov_w = sigma2_res * inv_XtX_reg

    # Return list of normal distributions for each coefficient
    return [
        GaussianDistribution(loc=mu_w[i], scale=np.sqrt(cov_w[i, i]))
        for i in range(n_features)
    ]


def gaussian_threshold(
    w: List[GaussianDistribution], threshold: float
) -> Tuple[List[GaussianDistribution], np.ndarray]:
    """
    Threshold coefficients: if |mean| < threshold, set to zero (i.e., delta at 0).
    Otherwise, keep the original distribution.
    """
    kept = []
    mask = np.zeros(len(w), dtype=bool)
    for i, wi in enumerate(w):
        if abs(wi.mean()) >= threshold:
            kept.append(wi)
            mask[i] = True
        else:
            # Replace with a degenerate normal at 0
            kept.append(
                GaussianDistribution(loc=0.0, scale=1e-10)
            )  # near-zero variance
            mask[i] = False
    return kept, mask


class STLSQl(pysindy.BaseOptimizer):
    """Sequentially thresholded least squares algorithm with Gaussian coefficients.

    Coefficients are represented as scipy.stats.norm objects.
    The target Y is assumed to be a Gaussian random variable.
    All operations follow standard Gaussian arithmetic.

    See the following reference for more details:

        Brunton, Steven L., Joshua L. Proctor, and J. Nathan Kutz.
        "Discovering governing equations from data by sparse
        identification of nonlinear dynamical systems."
        Proceedings of the national academy of sciences
        113.15 (2016): 3932-3937.
    """

    def __init__(
        self,
        threshold=0.1,
        alpha=0.05,
        max_iter=20,
        ridge_kw=None,
        normalize_columns=False,
        copy_X=True,
        initial_guess=None,
        verbose=False,
        sparse_ind=None,
        unbias=True,
    ):
        super().__init__(
            max_iter=max_iter,
            copy_X=copy_X,
            normalize_columns=normalize_columns,
            unbias=unbias,
        )

        if threshold < 0:
            raise ValueError("threshold cannot be negative")
        if alpha < 0:
            raise ValueError("alpha cannot be negative")

        self.threshold = threshold
        self.alpha = alpha
        self.ridge_kw = ridge_kw
        self.initial_guess = initial_guess
        self.verbose = verbose
        self.sparse_ind = sparse_ind

    def _sparse_coefficients(
        self, dim: int, ind_nonzero: np.ndarray, coef, threshold: float
    ):
        """Perform thresholding of the weight vector(s) (on specific indices
        if ``self.sparse_ind`` is not None)"""
        c = [GaussianDistribution(loc=0.0, scale=1e-10)] * dim
        for i, idx in enumerate(ind_nonzero):
            c[idx] = coef[i]
        big_ind = np.array([abs(w.mean()) >= threshold for w in c])
        if self.sparse_ind is not None:
            nonsparse_ind_mask = np.ones_like(ind_nonzero, dtype=bool)
            nonsparse_ind_mask[self.sparse_ind] = False
            big_ind = big_ind | nonsparse_ind_mask
        # Zero out small ones
        for i in range(len(c)):
            if not big_ind[i]:
                c[i] = GaussianDistribution(loc=0.0, scale=1e-10)
        return c, big_ind

    def _regress(self, x: np.ndarray, y: np.ndarray, dim: int, sparse_sub: np.ndarray):
        """Perform ridge regression with Gaussian output (coefficients as norm objects)."""
        kw = self.ridge_kw or {}
        # Use Gaussian ridge regression
        try:
            coef = gaussian_ridge_regression(x, y, self.alpha)
        except Exception as e:
            # Fallback: use deterministic ridge and assume small uncertainty
            coef = [GaussianDistribution(loc=0.0, scale=1e-10)] * dim
            warnings.warn(f"Ridge regression failed: {e}")
        return coef

    def _no_change(self) -> bool:
        """Check if the coefficient mask has changed after thresholding."""
        if len(self.history_) < 2:
            return False
        this_coef = self.history_[-1]
        last_coef = self.history_[-2]
        # Compare means (since variances may differ slightly)
        return all(
            abs(w1.mean() - w2.mean()) < 1e-8 for w1, w2 in zip(this_coef, last_coef)
        )

    def _reduce(self, x: np.ndarray, y: np.ndarray):
        """Performs at most ``self.max_iter`` iterations of the
        sequentially-thresholded least squares algorithm with Gaussian coefficients."""
        n_samples, n_features = x.shape
        n_targets = y.shape[1]

        # Initialize coefficients
        if self.initial_guess is not None:
            # Convert initial guess to list of norm objects
            if isinstance(self.initial_guess, np.ndarray):
                self.coef_ = [
                    GaussianDistribution(loc=mu, scale=1e-10)
                    for mu in self.initial_guess.flatten()
                ]
            else:
                self.coef_ = [
                    GaussianDistribution(loc=mu, scale=1e-10)
                    for mu in self.initial_guess
                ]
        else:
            # Use initial ridge regression
            self.coef_ = gaussian_ridge_regression(x, y, self.alpha)

        # Initialize support (ind_)
        self.ind_ = np.array([abs(w.mean()) >= self.threshold for w in self.coef_])

        # Initialize history
        self.history_ = [self.coef_[:]]

        # Print initial values
        if self.verbose:
            row = [
                "Iteration",
                "|y - Xw|^2",
                "a * |w|_2",
                "|w|_0",
                "Total error: |y - Xw|^2 + a * |w|_2",
            ]
            print(
                "{: >10} ... {: >10} ... {: >10} ... {: >10}"
                " ... {: >10}".format(*row)
            )

        for k in range(self.max_iter):
            if np.count_nonzero(self.ind_) == 0:
                warnings.warn(
                    "Sparsity parameter is too big ({}) and eliminated all "
                    "coefficients".format(self.threshold)
                )
                self.coef_ = [GaussianDistribution(loc=0.0, scale=1e-10)] * n_features
                break

            # Perform regression on current support
            optvar = []
            for i in range(n_targets):
                # Extract relevant columns
                breakpoint()
                x_i = x[:, self.ind_]
                y_i = y[:, i]

                # Perform ridge regression
                coef_i = self._regress(x_i, y_i, np.count_nonzero(self.ind_), None)

                # Apply thresholding
                coef_i, ind_i = self._sparse_coefficients(
                    n_features, np.where(self.ind_)[0], coef_i, self.threshold
                )
                optvar.append(coef_i)
                self.ind_[i] = ind_i

            # Update history
            self.history_.append(optvar[0] if n_targets == 1 else optvar)

            # Print progress
            if self.verbose:
                # Compute prediction error
                y_pred = np.array(
                    [gaussian_linear_combination(x, w).mean() for w in optvar]
                )
                R2 = np.sum((y - y_pred) ** 2)
                L2 = self.alpha * np.sum([w.var() for w in optvar[0]])
                L0 = np.count_nonzero([w.mean() != 0 for w in optvar[0]])
                row = [k, R2, L2, L0, R2 + L2]
                print(
                    "{0:10d} ... {1:10.4e} ... {2:10.4e} ... {3:10d}"
                    " ... {4:10.4e}".format(*row)
                )

            # Check convergence
            if np.sum(self.ind_) == np.count_nonzero(self.ind_) or self._no_change():
                break
        else:
            warnings.warn(
                "STLSQ did not converge after {self.max_iter} iterations.",
                ConvergenceWarning,
            )

        # Finalize coefficients
        self.coef_ = optvar[0] if n_targets == 1 else optvar

    @property
    def complexity(self):
        check_is_fitted(self)
        # Count non-zero mean coefficients
        return np.count_nonzero(
            [abs(w.mean()) >= self.threshold for w in self.coef_]
        ) + (1 if abs(self.intercept_) >= self.threshold else 0)

    def _unbias(self, x: np.ndarray, y: np.ndarray):
        if not self.sparse_ind:
            return super()._unbias(x, y)
        # Similar logic, but now with Gaussian coefficients
        # This is a placeholder — full implementation would recompute bias
        # using Gaussian regression on intercept
        pass


class STLSQ(pysindy.STLSQ):

    def _sparse_coefficients(
        self, dim: int, ind_nonzero: np.ndarray, coef: np.ndarray, threshold: float
    ) -> (np.ndarray, np.ndarray):
        """Perform thresholding of the weight vector(s) (on specific indices
        if ``self.sparse_ind`` is not None)"""
        c = np.zeros(dim)
        for i, idx in enumerate(ind_nonzero):
            c[idx] = coef[i]

        big_ind = np.array([_c.norm() >= threshold for _c in coef])
        if self.sparse_ind is not None:
            nonsparse_ind_mask = np.ones_like(ind_nonzero)
            nonsparse_ind_mask[self.sparse_ind] = False
            big_ind = big_ind | nonsparse_ind_mask
        c[~big_ind] = 0
        return c, big_ind

    def _regress(self, x: np.ndarray, y: np.ndarray, dim: int, sparse_sub: np.ndarray):
        """Perform the ridge regression (on specific indices if
        ``self.sparse_ind`` is not None)"""
        kw = self.ridge_kw or {}
        if self.sparse_ind is None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=LinAlgWarning)
                try:
                    coef = gaussian_ridge_regression(x, y, self.alpha, **kw)
                except LinAlgWarning:
                    # increase alpha until warning stops
                    self.alpha = 2 * self.alpha
            self.iters += 1
            return coef
        if self.sparse_ind is not None:
            alpha_array = np.zeros((dim, dim))
            alpha_array[sparse_sub, sparse_sub] = np.sqrt(self.alpha)
            x_lin = np.concatenate((x, alpha_array), axis=0)
            y_lin = np.concatenate((y, np.zeros((dim,))))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=LinAlgWarning)
                try:
                    coef = (
                        LinearRegression(fit_intercept=False, **kw)
                        .fit(x_lin, y_lin)
                        .coef_
                    )
                except LinAlgWarning:
                    # increase alpha until warning stops
                    self.alpha = 2 * self.alpha
            self.iters += 1
            return coef

    def _no_change(self):
        """Check if the coefficient mask has changed after thresholding"""
        this_coef = self.history_[-1].flatten()
        if len(self.history_) > 1:
            last_coef = self.history_[-2].flatten()
        else:
            last_coef = np.zeros_like(this_coef)
        return all(bool(i) == bool(j) for i, j in zip(this_coef, last_coef))

    def _reduce(self, x, y):
        """Performs at most ``self.max_iter`` iterations of the
        sequentially-thresholded least squares algorithm.

        Assumes an initial guess for coefficients and support are saved in
        ``self.coef_`` and ``self.ind_``.
        """
        if self.initial_guess is not None:
            self.coef_ = self.initial_guess

        ind = self.ind_
        n_samples, n_features = x.shape
        n_targets = y.shape[1]
        n_features_selected = np.sum(ind)
        sparse_sub = [np.array(self.sparse_ind)] * y.shape[1]
        optvar = np.zeros((n_targets, n_features))

        # Print initial values for each term in the optimization
        if self.verbose:
            row = [
                "Iteration",
                "|y - Xw|^2",
                "a * |w|_2",
                "|w|_0",
                "Total error: |y - Xw|^2 + a * |w|_2",
            ]
            print(
                "{: >10} ... {: >10} ... {: >10} ... {: >10}"
                " ... {: >10}".format(*row)
            )

        for k in range(self.max_iter):
            if np.count_nonzero(ind) == 0:
                warnings.warn(
                    "Sparsity parameter is too big ({}) and eliminated all "
                    "coefficients".format(self.threshold)
                )
                optvar = np.zeros((n_targets, n_features))
                break

            optvar = np.zeros((n_targets, n_features))
            for i in range(n_targets):
                if np.count_nonzero(ind[i]) == 0:
                    warnings.warn(
                        "Sparsity parameter is too big ({}) and eliminated all "
                        "coefficients".format(self.threshold)
                    )
                    continue
                coef_i = self._regress(
                    x[:, ind[i]], y[:, i], np.count_nonzero(ind[i]), sparse_sub[i]
                )
                coef_i, ind_i = self._sparse_coefficients(
                    n_features, ind[i], coef_i, self.threshold
                )
                if self.sparse_ind is not None:
                    vals_to_remove = np.intersect1d(
                        self.sparse_ind, np.where(coef_i == 0)
                    )
                    sparse_sub[i] = _remove_and_decrement(
                        self.sparse_ind, vals_to_remove
                    )
                optvar[i] = coef_i
                ind[i] = ind_i

            self.history_.append(optvar)
            if self.verbose:
                R2 = np.sum((y - np.dot(x, optvar.T)) ** 2)
                L2 = self.alpha * np.sum(optvar**2)
                L0 = np.count_nonzero(optvar)
                row = [k, R2, L2, L0, R2 + L2]
                print(
                    "{0:10d} ... {1:10.4e} ... {2:10.4e} ... {3:10d}"
                    " ... {4:10.4e}".format(*row)
                )
            if np.sum(ind) == n_features_selected or self._no_change():
                # could not (further) select important features
                break
        else:
            warnings.warn(
                "STLSQ did not converge after {self.max_iter} iterations.",
                ConvergenceWarning,
            )
        if self.sparse_ind is None:
            self.coef_ = optvar
            self.ind_ = ind
        else:
            non_sparse_ind = np.setxor1d(self.sparse_ind, range(n_features))
            self.coef_ = optvar[:, self.sparse_ind]
            self.ind_ = ind[:, self.sparse_ind]
            self.optvar_non_sparse_ = optvar[:, non_sparse_ind]
            self.ind_non_sparse_ = ind[:, non_sparse_ind]

    @property
    def complexity(self):
        check_is_fitted(self)

        return np.count_nonzero(self.coef_) + np.count_nonzero(
            [abs(self.intercept_) >= self.threshold]
        )

    def _unbias(self, x: np.ndarray, y: np.ndarray):
        if not self.sparse_ind:
            return super()._unbias(x, y)
        regression_col_mask = np.zeros((y.shape[1], x.shape[1]), dtype=bool)
        regression_col_mask[:, self.sparse_ind] = self.ind_
        non_sparse_ind = np.setxor1d(self.sparse_ind, range(x.shape[1]))
        regression_col_mask[:, non_sparse_ind] = self.ind_non_sparse_

        for i in range(self.ind_.shape[0]):
            if np.any(self.ind_[i]):
                optvar = (
                    LinearRegression(fit_intercept=False)
                    .fit(x[:, regression_col_mask[i]], y[:, i])
                    .coef_
                )
                self.coef_[i] = optvar[self.sparse_ind]
                self.optvar_non_sparse_[i] = optvar[non_sparse_ind]
