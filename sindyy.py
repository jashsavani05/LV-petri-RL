#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt
from lv_sim import LV_SIM
import pysindy as ps
from tqdm import tqdm
from scipy.integrate import odeint
import pygad
from collections import namedtuple


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
        S=np.array([[-1, 0, 1], [1, -1, 0]]),  # Interaction matrix
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
        self.gradient_descent_step = gradient_descent_step

        _f = self.feature_library.fit(np.zeros_like((1, self.spn_model_target_dim)))

        if self.W is None:
            self.W = np.ones((self.S.shape[1], len(_f.get_feature_names())))

        self.P_ = self.feature_library.fit(
            np.zeros_like((1, self.spn_model_target_dim))
        )

    def loss_and_grad(self, Ys, ts, W):
        Ps = [self.P_.transform(y) for y in Ys]
        Y_diff_preds = [self.S @ W @ P for P in Ps]
        losses = []
        loss_grads = []
        N = len(Ys)
        for i in range(1, len(Ys)):
            Y_i, Y_im1 = (Ys[i], Ys[i - 1])
            dY_i_pred, dY_im1_pred = (Y_diff_preds[i], Y_diff_preds[i - 1])
            t_i, t_im1 = (ts[i], ts[i - 1])
            P_diff = (dY_i_pred - dY_im1_pred) / 2
            C = (Y_i - Y_im1) / (t_i - t_im1)
            K = P_diff - C
            loss = (sum(K**2)) / 2
            grad_loss = self.S.T @ K @ P_diff.T
            losses.append(loss)
            loss_grads.append(grad_loss)

        loss = sum(losses) / N
        grad_loss = sum(loss_grads) / N

        return loss, grad_loss

    # def on_gen(self, ga_instance):
    #     # ga_instance.plot_fitness()
    #     pass

    # def run_genetic(
    #     self,
    #     x0s,
    #     Ys,
    #     t_intervals,
    #     num_generations=30,
    #     num_parents_mating=5,
    #     sol_per_pop=10,
    #     init_range_low=-2,
    #     init_range_high=5,
    #     parent_selection_type="sss",
    #     keep_parents=1,
    #     crossover_type="single_point",
    #     mutation_percent_genes=10,
    #     mutation_type="random",
    #     plot=False,
    # ):
    #     def fitness_func(ga_instance, solution, solution_idx):
    #         W = solution.reshape(*self.W.shape)
    #         Y_preds = self.propagate_(x0s, t_intervals, W)
    #         loss = self.loss_(Y_preds, Ys, W)
    #         return -loss

    #     ga_instance = pygad.GA(
    #         fitness_func=fitness_func,
    #         on_generation=self.on_gen,
    #         num_generations=num_generations,
    #         num_genes=self.W.shape[0] * self.W.shape[1],
    #         num_parents_mating=num_parents_mating,
    #         sol_per_pop=sol_per_pop,
    #         init_range_low=init_range_low,
    #         init_range_high=init_range_high,
    #         parent_selection_type=parent_selection_type,
    #         keep_parents=keep_parents,
    #         crossover_type=crossover_type,
    #         mutation_percent_genes=mutation_percent_genes,
    #         mutation_type=mutation_type,
    #     )

    #     ga_instance.run()
    #     return ga_instance

    def fit(self, Ys, ts, max_iterations=100, plot=False):
        W = self.W
        losses = []
        for it in tqdm(range(max_iterations)):
            loss, grad = self.loss_and_grad(Ys, ts, W)
            losses.append(loss)
            if plot and (it % 10 == 0):
                plt.plot(list(range(it + 1)), losses)
                plt.show()
            losses.append(loss)
            W = W - self.gradient_descent_step * grad
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
                p = self.P_.transform(np.array([vec]))
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

    model = model.fit(Ys, t, plot=True)
    print(model.W)
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
    return model


model = run()
