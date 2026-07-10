#!/usr/bin/env python3

import os
import random
from pprint import pprint

# Import and activate the plugin (and the rest of SNAKES)
import snakes.plugins
import snakes_spn.plugin as spn_plugin
from snakes.typing import tAll
from snakes.nets import *
import numpy as np

snakes.plugins.load([spn_plugin, "gv"], "snakes.nets", "snk")

import matplotlib.pyplot as plt

from snk import (
    PetriNet,
    Place,
    Transition,
    tInteger,
    tFloat,
    Expression,
    Variable,
    Instance,
)

from spn_tools.run_simulation import run_simulation


def setup_lotka_volterra_simulation():
    """TODO describe function

    :returns:

    """
    lv_cpn = PetriNet("Lotka-Volterra CPN Model")
    alpha, beta, gamma, K = 0.5, 0.3, 0.002, 350
    lv_cpn.globals["alpha"] = alpha
    lv_cpn.globals["beta"] = beta
    lv_cpn.globals["gamma"] = gamma
    lv_cpn.globals["K"] = K

    p_predator = Place("Predator", 50)
    p_prey = Place("Prey", 100)
    lv_cpn.add_place(p_predator)
    lv_cpn.add_place(p_prey)

    exp_predator_huntrate = Expression("gamma * x * y")
    exp_predator_huntrate.globals["gamma"] = gamma
    t_predator_born = Transition(
        name="Predator hunts prey", rate_function=exp_predator_huntrate
    )
    lv_cpn.add_transition(t_predator_born)

    lv_cpn.add_input(p_predator.name, t_predator_born.name, Variable("y"))
    lv_cpn.add_input(p_prey.name, t_predator_born.name, Variable("x"))
    lv_cpn.add_output(p_predator.name, t_predator_born.name, Expression("y + 1"))
    lv_cpn.add_output(p_prey.name, t_predator_born.name, Expression("x - 1"))

    exp_predator_deathrate = Expression("beta * y")
    exp_predator_deathrate.globals["beta"] = beta
    t_predator_dies = Transition("Predator dies", rate_function=exp_predator_deathrate)
    lv_cpn.add_transition(t_predator_dies)

    lv_cpn.add_input(p_predator.name, t_predator_dies.name, Variable("y"))
    lv_cpn.add_output(p_predator.name, t_predator_dies.name, Expression("y - 1"))

    exp_prey_birthrate = Expression("alpha * x")

    exp_prey_birthrate.globals["alpha"] = alpha
    exp_prey_birthrate.globals["K"] = K
    t_prey_born = Transition("Prey born", rate_function=exp_prey_birthrate)
    lv_cpn.add_transition(t_prey_born)

    lv_cpn.add_input(p_prey.name, t_prey_born.name, Variable("x"))
    lv_cpn.add_output(p_prey.name, t_prey_born.name, Expression("x + 1"))

    return lv_cpn


def run_sim(lv_cpn, max_steps=20000, seed=0, plot=False):
    lv_cpn.draw("cpn.png")
    sim = run_simulation(
        lv_cpn, max_steps=max_steps, rng=np.random.default_rng(seed).random
    )

    if plot:
        plt.plot(sim["time"], sim["Predator"])
        plt.plot(sim["time"], sim["Prey"])
        plt.show()
    return sim
