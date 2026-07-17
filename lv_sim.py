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


class LV_SIM:
    def __init__(
        self,
        x0=(50, 150),
        expression_functions={
            "hunt_rate": (lambda x, y: 0.002 * x * y),
            "predator_death_rate": (lambda y: 0.3 * y),
            "prey_birth_rate": (lambda x: 0.5 * x),
        },
    ):
        self.x0 = x0
        self.lv_cpn = PetriNet("Lotka-Volterra CPN Model")
        p_predator = Place("Predator", x0[0])
        p_prey = Place("Prey", x0[1])
        self.lv_cpn.add_place(p_predator)
        self.lv_cpn.add_place(p_prey)

        exp_predator_huntrate = Expression("abs(hunt_rate(x, y))")
        exp_predator_huntrate.globals["hunt_rate"] = expression_functions["hunt_rate"]
        t_predator_born = Transition(
            name="Predator hunts prey", rate_function=exp_predator_huntrate
        )
        self.lv_cpn.add_transition(t_predator_born)

        self.lv_cpn.add_input(p_predator.name, t_predator_born.name, Variable("y"))
        self.lv_cpn.add_input(p_prey.name, t_predator_born.name, Variable("x"))
        self.lv_cpn.add_output(
            p_predator.name, t_predator_born.name, Expression("y + 1")
        )
        self.lv_cpn.add_output(p_prey.name, t_predator_born.name, Expression("x - 1"))

        exp_predator_deathrate = Expression("abs(predator_death_rate(y))")
        exp_predator_deathrate.globals["predator_death_rate"] = expression_functions[
            "predator_death_rate"
        ]
        t_predator_dies = Transition(
            "Predator dies", rate_function=exp_predator_deathrate
        )
        self.lv_cpn.add_transition(t_predator_dies)

        self.lv_cpn.add_input(p_predator.name, t_predator_dies.name, Variable("y"))
        self.lv_cpn.add_output(
            p_predator.name, t_predator_dies.name, Expression("y - 1")
        )

        exp_prey_birthrate = Expression("abs(prey_birth_rate(X))")

        exp_prey_birthrate.globals["prey_birth_rate"] = expression_functions[
            "prey_birth_rate"
        ]
        t_prey_born = Transition("Prey born", rate_function=exp_prey_birthrate)
        self.lv_cpn.add_transition(t_prey_born)
        self.lv_cpn.add_input(p_prey.name, t_prey_born.name, Variable("X"))
        self.lv_cpn.add_output(p_prey.name, t_prey_born.name, Expression("X + 1"))

    def run_sim(self, max_time=300, max_steps=10000, seed=0, plot=False):
        random.seed(0)
        sim = run_simulation(
            self.lv_cpn,
            max_steps=max_steps,
            max_time=max_time,
            rng=random.random,
        )

        if plot:
            plt.plot(sim["time"], sim["Predator"])
            plt.plot(sim["time"], sim["Prey"])
        return sim, ("Predator", "Prey"), "time"
