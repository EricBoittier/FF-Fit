from unittest import TestCase
import json

from ff_energy.simulations.templater import SimTemplate, config_files

class TestSimTemplate(TestCase):

    def test_sim_template(self):

        with open(config_files / "kmdcm.json") as f:
            kwargs = json.load(f)

        s = SimTemplate(kwargs)
        # print(s.get_sim())
        # with open("test_sim.inp", "w") as f:
        #     f.write(s.get_sim())
        for k, v in s.kwargs.items():
            print(k,":", v)

        print(s.get_submit())

        # print(s.__dict__)
