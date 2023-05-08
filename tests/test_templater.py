from unittest import TestCase
import json

from ff_energy.simulations.templater import SimTemplate

class TestSimTemplate(TestCase):

    def test_sim_template(self):

        with open("/Users/ericboittier/Documents/github/phd/ff_energy/"
              "ff_energy/simulations/configs/sim_test.json") as f:
            kwargs = json.load(f)

        s = SimTemplate(kwargs)
        print(s.get_sim())
        with open("test_sim.inp", "w") as f:
            f.write(s.get_sim())
        # print(s.__dict__)




