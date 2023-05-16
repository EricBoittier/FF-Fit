from unittest import TestCase
import json

from ff_energy.simulations.templater import SimTemplate, config_files
from ff_energy.simulations.sim import Simulation

temperatures = [300, 298]


def make_temp(sim):
    for t in temperatures:
        print(sim.basepathname)
        sim.set_T(t)
        sim.make()


class TestSimTemplate(TestCase):

    def test_sim_template(self):
        with open(config_files / "kmdcm.json") as f:
            kwargs = json.load(f)
        s = SimTemplate(kwargs)
        for k, v in s.kwargs.items():
            print(k, ":", v)
        print(s.get_submit())

    def test_sim_kmdcm(self):
        with open(config_files / "kmdcm.json") as f:
            kwargs = json.load(f)

        sim = Simulation(**kwargs)
        print(sim.basepathname)
        make_temp(sim)

    def test_sim_shake(self):
        with open(config_files / "shake.json") as f:
            kwargs = json.load(f)

        sim = Simulation(**kwargs)
        print(sim.basepathname)
        make_temp(sim)

    def test_sim_mdcm(self):
        with open(config_files / "mdcm.json") as f:
            kwargs = json.load(f)

        sim = Simulation(**kwargs)
        print(sim.basepathname)
        make_temp(sim)

    def test_sim_optpc(self):
        with open(config_files / "optpc.json") as f:
            kwargs = json.load(f)

        sim = Simulation(**kwargs)
        print(sim.basepathname)
        make_temp(sim)
