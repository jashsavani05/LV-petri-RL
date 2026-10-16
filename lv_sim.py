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
        # Balanced initial populations to stabilize the stochastic process
        x0=(100, 200, 30), 
        expression_functions={
            "prey_birth_rate": (lambda x: 0.6 * x),
            "hunt_rate": (lambda x, y: 0.003 * x * y),
            "predator_death_rate": (lambda y: 0.4 * y),
            # Fine-tuned to prevent over-hunting and produce clear cyclic waves
            "super_hunt_rate": (lambda x, y, z: 0.005 * x * y * z),
            "super_death_rate": (lambda z: 0.25 * z),
        },
    ):
        self.x0 = x0
        self.lv_cpn = PetriNet("Lotka-Volterra CPN Model")
        
        # Define Places
        p_predator = Place("Predator", x0[0])
        p_prey = Place("Prey", x0[1])
        p_super = Place("Superpredator", x0[2])
        
        self.lv_cpn.add_place(p_predator)
        self.lv_cpn.add_place(p_prey)
        self.lv_cpn.add_place(p_super)

        # --- Transition: Predator hunts prey ---
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

        # --- Transition: Predator dies ---
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

        # --- Transition: Prey born ---
        exp_prey_birthrate = Expression("abs(prey_birth_rate(X))")
        exp_prey_birthrate.globals["prey_birth_rate"] = expression_functions[
            "prey_birth_rate"
        ]
        t_prey_born = Transition("Prey born", rate_function=exp_prey_birthrate)
        self.lv_cpn.add_transition(t_prey_born)
        self.lv_cpn.add_input(p_prey.name, t_prey_born.name, Variable("X"))
        self.lv_cpn.add_output(p_prey.name, t_prey_born.name, Expression("X + 1"))

        # --- Transition: Superpredator hunts both Prey and Predator ---
        exp_super_huntrate = Expression("abs(super_hunt_rate(x, y, z))")
        exp_super_huntrate.globals["super_hunt_rate"] = expression_functions["super_hunt_rate"]
        t_super_hunts = Transition(
            name="Superpredator hunts", rate_function=exp_super_huntrate
        )
        self.lv_cpn.add_transition(t_super_hunts)
        
        self.lv_cpn.add_input(p_prey.name, t_super_hunts.name, Variable("x"))
        self.lv_cpn.add_input(p_predator.name, t_super_hunts.name, Variable("y"))
        self.lv_cpn.add_input(p_super.name, t_super_hunts.name, Variable("z"))
        
        self.lv_cpn.add_output(p_prey.name, t_super_hunts.name, Expression("x - 1"))
        self.lv_cpn.add_output(p_predator.name, t_super_hunts.name, Expression("y - 1"))
        self.lv_cpn.add_output(p_super.name, t_super_hunts.name, Expression("z + 1"))

        # --- Transition: Superpredator dies ---
        exp_super_deathrate = Expression("abs(super_death_rate(z))")
        exp_super_deathrate.globals["super_death_rate"] = expression_functions["super_death_rate"]
        t_super_dies = Transition(
            name="Superpredator dies", rate_function=exp_super_deathrate
        )
        self.lv_cpn.add_transition(t_super_dies)
        
        self.lv_cpn.add_input(p_super.name, t_super_dies.name, Variable("z"))
        self.lv_cpn.add_output(p_super.name, t_super_dies.name, Expression("z - 1"))

    def run_sim(self, max_time=300, max_steps=10000, seed=0, plot=False):
        random.seed(seed)
        sim = run_simulation(self.lv_cpn, max_steps=max_steps, max_time=max_time)

        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(sim["time"], sim["Prey"], label="Prey", color="green")
            plt.plot(sim["time"], sim["Predator"], label="Predator", color="blue")
            plt.plot(sim["time"], sim["Superpredator"], label="Superpredator", color="purple")
            plt.xlabel("Time")
            plt.ylabel("Population")
            plt.title("Three-Species Lotka-Volterra CPN Simulation")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            
            # Save visual outputs
            plt.savefig("simulation_graph.png", dpi=300)
            print("Simulation plot saved successfully as 'simulation_graph.png'")
            
            try:
                self.lv_cpn.draw("cpn.png")
                print("Petri net schematic saved successfully as 'cpn.png'")
            except Exception as e:
                print(f"Structural rendering note: {e}")
                
        return sim, ("Predator", "Prey", "Superpredator"), "time"


if __name__ == "__main__":
    scpn = LV_SIM()
    sim = scpn.run_sim(plot=True)
    plt.show()
