#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt
from lv_sim import LV_SIM
import pysindy as ps
from tqdm import tqdm
from scipy.integrate import odeint
import pygad


class STLSQ:
    def __init__(
        self,
        spn_model=LV_SIM,
        spn_model_target_dim=2,
        spn_model_expressions={
            "hunt_rate": [True, True],
            "predator_death_rate": [False, True],
            "prey_birth_rate": [True, False],
        },
        S=np.array([[-1, 0, 1], [1, -1, 0]]),  # Interaction function
        train_weights=None,
        feature_library=ps.CustomLibrary(
            library_functions=[
                lambda x: x,
                lambda x: x * x,
                lambda x, y: x * y,
            ]
        ),
        threshold=0.0001,
        gradient_descent_step=0.01,
        integration_samples=4,  # large values increase integration accuracy, but also time complexity
        initial_guess=None,
        sparse_ind=None,  # should have same shape as initial_guess
    ):
        self.spn_model = spn_model
        self.spn_model_target_dim = spn_model_target_dim
        self.spn_model_targets = spn_model_expressions
        self.S = S
        self.feature_library = feature_library
        self.spn_model_expressions = spn_model_expressions
        self.threshold = threshold
        self.sparse_ind = sparse_ind
        self.train_weights = train_weights
        self.W = initial_guess
        self.integration_samples = integration_samples
        self.gradient_descent_step = gradient_descent_step

        _f = self.feature_library.fit(np.zeros_like((1, self.spn_model_target_dim)))

        if self.W is None:
            self.W = np.ones((self.S.shape[1], len(_f.get_feature_names())))

        self.P_ = [
            self.feature_library.fit(np.zeros_like((1, self.spn_model_target_dim)))
        ]

    def step(self, x0, t, W, dt):
        p_ = np.array([func.transform(np.array([x0])) for func in self.P_])
        s_ = ((self.S @ W @ p_[0].T) * dt).T[0]
        if np.isnan(p_).any():
            breakpoint()
        return ((self.S @ W @ p_[0].T) * dt).T[0]

    def step_p(self, p0, t, x0, dt):
        p_ = self.P_[0].transform(np.array([x0]))
        return (p_ * dt)[0][0]

    def propagate_ps_(self, x0s, t_intervals):
        p = np.array(
            [
                odeint(
                    self.step_p,
                    (self.P_[0].transform(x0) * 0)[0],
                    np.linspace(t0, t1, self.integration_samples),
                    args=(x0, t1 - t0),
                )[-1]
                for x0, (t0, t1) in zip(x0s, t_intervals)
            ]
        )
        if np.isnan(p).any():
            breakpoint()
        return p

    # def integrate(self, initial_condition, )

    def propagate_(self, x0s, t_intervals, W):

        Y_preds = np.array(
            [
                odeint(
                    self.step,
                    x0[0],
                    np.linspace(t0, t1, self.integration_samples),
                    args=(W, (t1 - t0)),
                )[-1]
                for x0, (t0, t1) in tqdm(zip(x0s, t_intervals))
            ]
        )
        if np.isnan(Y_preds).any():
            breakpoint()
        return Y_preds

    def residuals_(self, Y_preds, Ys):
        return abs(Ys - Y_preds)

    def loss_(self, Y_preds, Ys, W, residuals=None):
        if residuals is None:
            residuals = self.residuals_(Y_preds, Ys)
        return 0.5 * (
            (residuals**2).sum() / (self.integration_samples * (len(Ys)))
        )  # + (W**2).sum()

    def grad_loss_(self, Y_preds, Ys, x0s, t_intervals, W, residuals=None):
        if residuals is None:
            residuals = self.residuals_(Y_preds, Ys)
        int_ps = self.propagate_ps_(x0s, t_intervals)
        zP = None
        for z_t, p_t in zip(residuals, int_ps):
            z_t = np.array([z_t])
            p_t = np.array([p_t])
            if zP is None:
                zP = z_t.T @ np.ones_like(p_t)
            else:
                zP = zP + z_t.T @ np.ones_like(p_t)
            if np.isnan(zP).any():
                breakpoint()
        return self.S.T @ zP  # + abs(W)

    def on_gen(self, ga_instance):
        #ga_instance.plot_fitness()
        pass

    def run_genetic(
        self,
        x0s,
        Ys,
        t_intervals,
        num_generations=30,
        num_parents_mating=5,
        sol_per_pop=10,
        init_range_low=-2,
        init_range_high=5,
        parent_selection_type="sss",
        keep_parents=1,
        crossover_type="single_point",
        mutation_percent_genes=10,
        mutation_type="random",
        plot=False,
    ):
        def fitness_func(ga_instance, solution, solution_idx):
            W = solution.reshape(*self.W.shape)
            Y_preds = self.propagate_(x0s, t_intervals, W)
            loss = self.loss_(Y_preds, Ys, W)
            return -loss

        ga_instance = pygad.GA(
            fitness_func=fitness_func,
            on_generation=self.on_gen,
            num_generations=num_generations,
            num_genes=self.W.shape[0] * self.W.shape[1],
            num_parents_mating=num_parents_mating,
            sol_per_pop=sol_per_pop,
            init_range_low=init_range_low,
            init_range_high=init_range_high,
            parent_selection_type=parent_selection_type,
            keep_parents=keep_parents,
            crossover_type=crossover_type,
            mutation_percent_genes=mutation_percent_genes,
            mutation_type=mutation_type,
        )

        ga_instance.run()
        return ga_instance

    def fit(self, x0s, Ys, t_intervals, max_iterations=10, plot=False):
        W = self.W
        losses = []
        for it in tqdm(range(max_iterations)):
            Y_preds = self.propagate_(x0s, t_intervals, W)
            ts = [t0 for t0, t1 in t_intervals]
            Y_preds_preys = [y_pr[0] for y_pr in Y_preds]
            Y_preds_preds = [y_pr[1] for y_pr in Y_preds]
            Y_preys = [y_pr[0] for y_pr in Ys]
            Y_predas = [y_pr[1] for y_pr in Ys]
            if plot:
                plt.plot(ts, Y_preds_preys, linestyle="--", color="black")
                plt.plot(ts, Y_preds_preds, linestyle="--", color="blue")
                plt.plot(ts, Y_preys, color="black")
                plt.plot(ts, Y_predas, color="blue")
                plt.show()
            residuals = self.residuals_(Y_preds, Ys)
            loss = self.loss_(Y_preds, Ys, W, residuals=residuals)
            losses.append(loss)
            print(loss)
            grad_l = self.grad_loss_(
                Y_preds, Ys, x0s, t_intervals, W, residuals=residuals
            )
            print(W)
            W = W - self.gradient_descent_step * grad_l / (
                self.integration_samples * (len(Ys))
            )
            if plot and (it % 100 == 0):
                plt.plot(list(range(it + 1)), losses)
                plt.show()
        self.W = W

        return self

    def new_spn(self, x0):
        expression_functions = {}

        for i, expf in enumerate(list(self.spn_model_expressions.keys())):

            def func(*args):
                vec = []
                arguments = list(args)
                for ind in self.spn_model_expressions[expf]:
                    if ind:
                        vec.append(arguments.pop())
                    else:
                        vec.append(0)
                p = self.P_[0].transform(np.array([vec]))
                rate = p @ self.W[i].reshape(p.shape[1], 1)
                return (p @ self.W[i].reshape(p.shape[1], 1))[0][0]

            expression_functions[expf] = func
        lv_cpn = LV_SIM(x0=x0, expression_functions=expression_functions)
        return lv_cpn


def run():
    x0 = (50, 150)
    lv_cpn = LV_SIM(x0=x0)
    sim, _, _ = lv_cpn.run_sim(max_steps=20000, seed=0, plot=True)
    indices = list(range(len(sim["time"])))[0::500]
    X = np.stack(
        (np.take(sim["Prey"], indices), np.take(sim["Predator"], indices)), axis=-1
    )
    t = np.array(np.take(sim["time"], indices))
    plt.scatter(t, np.take(sim["Predator"], indices), marker="x", color="black")
    plt.scatter(t, np.take(sim["Prey"], indices), marker="x", color="black")
    plt.show()
    x0s, Ys, t_intervals = [], [], []
    for i in range(len(t) - 1):
        x0s.append(np.array([X[i]]))
        Ys.append(X[i + 1])
        t_intervals.append((t[i], t[i + 1]))
    model = STLSQ()
    Ys = np.array(Ys)
    ga_instance = model.run_genetic(x0s, Ys, t_intervals, num_generations=30, plot=True)
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    W = solution.reshape(3, 5)
    model.W = W
    print(W)
    lv_cpn2 = model.new_spn(x0)
    sim2, _, _ = lv_cpn2.run_sim(max_steps=20000, seed=0, plot=False)
    plt.plot(sim["time"], sim["Prey"], color="black")
    plt.plot(sim["time"], sim["Predator"], color="orange")
    plt.scatter(t, np.take(sim["Predator"], indices), marker="x", color="black")
    plt.scatter(t, np.take(sim["Prey"], indices), marker="x", color="black")
    plt.plot(sim2["time"], sim2["Prey"], linestyle="--", color="black")
    plt.plot(sim2["time"], sim2["Predator"], linestyle="--", color="orange")
    plt.scatter(t, np.take(sim["Predator"], indices), marker="x", color="black")
    plt.scatter(t, np.take(sim["Prey"], indices), marker="x", color="black")
    plt.show()
    ga_instance.plot_fitness()
    return ga_instance


run()
