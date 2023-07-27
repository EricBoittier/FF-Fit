import os
import sys
import numpy as np
import pandas as pd

import pycharmm
import pycharmm.generate as gen
import pycharmm.ic as ic
import pycharmm.coor as coor
import pycharmm.energy as energy
import pycharmm.dynamics as dyn
import pycharmm.nbonds as nbonds
import pycharmm.minimize as minimize
import pycharmm.crystal as crystal
import pycharmm.select as select
import pycharmm.image as image
import pycharmm.psf as psf
import pycharmm.param as param
import pycharmm.read as read
import pycharmm.write as write
import pycharmm.settings as settings
import pycharmm.cons_harm as cons_harm
import pycharmm.cons_fix as cons_fix
import pycharmm.shake as shake
import pycharmm.scalar as scalar

from ff_energy.pycharmm_runner.nbonds_dicts import DEFAULT_NBONDS_DICT
from ff_energy.pycharmm_runner.mini_dicts import ABNR_DICT
from ff_energy.pycharmm_runner.input_dicts import DCM_TEST
from ff_energy.pycharmm_runner.dyna_dicts import get_dynamics_dict



def nbonds_setup(dict=None):
    if dict is None:
        pycharmm.NonBondedScript(**DEFAULT_NBONDS_DICT)
    else:
        pycharmm.NonBondedScript(**dict)

class PyCHARMM_Runner:
    def __init__(self, kwargs):
        # set the default values (pdb name, toppar stream, sequence)
        if "pdb_name" not in kwargs:
            raise ValueError("pdb_name must be specified")
        else:
            self.pdb_name = kwargs["pdb_name"]
        if "toppar_stream" not in kwargs:
            raise ValueError("toppar_stream must be specified")
        else:
            self.toppar_stream = kwargs["toppar_stream"]
        if "sequence" not in kwargs:
            raise ValueError("sequence must be specified")
        else:
            self.sequence = kwargs["sequence"]
        if "sidelength" not in kwargs:
            raise ValueError("sidelength must be specified")
        else:
            self.sidelength = kwargs["sidelength"]
        # image cutoff
        if "image_cutoff" not in kwargs:
            self.image_cutoff = 12.0
        else:
            self.image_cutoff = kwargs["image_cutoff"]
        if "output_path" not in kwargs:
            self.output_path = "output"
        else:
            self.output_path = kwargs["output_path"]

    def __repr__(self):
        rep_str = "".join([f"{key}: {value}\n"
                           for key, value in self.__dict__.items()])
        return rep_str

    def setup(self):
        pycharmm.lingo.charmm_script('bomlev 0')
        pycharmm.lingo.charmm_script(self.toppar_stream)
        read.sequence_string(self.sequence)
        gen.new_segment(
            seg_name='X',
            setup_ic=True)
        read.pdb(self.pdb_name)

    def imageinit(self, side_length=None, cutoff=None):
        if side_length is None:
            side_length = self.sidelength
        if cutoff is None:
            cutoff = self.image_cutoff

        crystal.define_cubic(side_length)
        crystal.build(cutoff)

    def minimize(self, dict=None, type="abnr"):
        if type == "abnr":
            if dict is None:
                dict = ABNR_DICT
            minimize.run_abnr(**dict)

    def dynamics(self,
                 dict=None,
                 type=None,
                 resname="heat",
                 nstep=10000,
                 timestep=0.0005):
        res_file, dcd_file = self.write_res_dcd(resname)
        dynamics_dict = get_dynamics_dict(timestep, res_file, dcd_file)
        dyn_heat = pycharmm.DynamicsScript(**dynamics_dict)
        dyn_heat.run()
        res_file.close()
        dcd_file.close()

    def write_res_dcd(self, resname):
        res_file = pycharmm.CharmmFile(
            file_name=f'{self.output_path}/{resname}.res', file_unit=2, formatted=True, read_only=False)
        dcd_file = pycharmm.CharmmFile(
            file_name=f'{self.output_path}/{resname}.dcd', file_unit=1, formatted=False, read_only=False)
        return res_file, dcd_file

    def write_info(self, key, note=""):
        write.coor_pdb(f"{self.output_path}/{key}.pdb", title=f"{key}: {note}")
        write.psf_card(f"{self.output_path}/{key}.psf", title=f"{key}: {note}")

    def routine(self):
        self.setup()
        self.imageinit()
        self.minimize()
        self.write_info("dcm_box", note="dcm_box")
        self.dynamics()
        self.write_info("dcm_box_heat", note="dcm_box_heat")


if __name__ == "__main__":
    pcr = PyCHARMM_Runner(DCM_TEST)
    print(pcr)
    pcr.routine()


