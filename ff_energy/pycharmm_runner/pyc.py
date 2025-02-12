import pycharmm
import pycharmm.generate as gen
import pycharmm.minimize as minimize
import pycharmm.crystal as crystal
import pycharmm.read as read
import pycharmm.write as write
import pycharmm.image as image

from ff_energy.pycharmm_runner.nbonds_dicts import DEFAULT_NBONDS_DICT
from ff_energy.pycharmm_runner.mini_dicts import ABNR_DICT
from ff_energy.pycharmm_runner.input_dicts import DCM_TEST
from ff_energy.pycharmm_runner.dyna_dicts import get_dynamics_dict
from ff_energy.pycharmm_runner.ase_helper import get_pmass


def nbonds_setup(dict=None):
    if dict is None:
        pycharmm.NonBondedScript(**DEFAULT_NBONDS_DICT)
    else:
        pycharmm.NonBondedScript(**dict)


class PyCHARMM_Runner:
    def __init__(self, kwargs):
        self.load_dcm = False

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
        if "dcm_file_path" not in kwargs:
            self.dcm_file_path = None
        else:
            self.dcm_file_path = kwargs["dcm_file_path"]
            self.load_dcm = True

    def __repr__(self):
        rep_str = "".join([f"{key}: {value}\n" for key, value in self.__dict__.items()])
        return rep_str

    def setup(self):
        """Set up the system by reading a toppar stream file and build the
        system from the sequence, then read in the pdb file for coordinates
        """
        pycharmm.lingo.charmm_script("bomlev 0")
        pycharmm.lingo.charmm_script(self.toppar_stream)
        read.sequence_string(self.sequence)
        gen.new_segment(seg_name="X", setup_ic=True)
        read.pdb(self.pdb_name)

    def dcm_setup(self, dump=False):
        """Uses the charmm linguo utility to run DCM"""
        _str = """! DCM SETUP ! the xyz file to dump coords!
{OPEN_DUMP}
open unit 10 card read name {DCM_FILE_PATH}
! start the DCM module
DCM IUDCM 10 TSHIFT {LOAD_DUMP}
! close the dcm input file
close unit 10"""
        #  If the coordinates will be saved to a dump file, then
        if dump:
            open_dump = "open unit 15 write card name dump.xyz"
            load_dump = "XYZ 15"
        else:
            open_dump = ""
            load_dump = ""

        _str = _str.format(
            DCM_FILE_PATH=self.dcm_file_path, OPEN_DUMP=open_dump, LOAD_DUMP=load_dump
        )
        pycharmm.lingo.charmm_script(_str)

    def imageinit(self, side_length=None, cutoff=None):
        """Set up the cubic periodic boundary conditions

        :param side_length: None or float
        :param cutoff: None or float
        :return: None
        """
        if side_length is None:
            side_length = self.sidelength
        if cutoff is None:
            cutoff = self.image_cutoff

        crystal.define_cubic(side_length)
        crystal.build(cutoff)

        # CHARMM scripting: image byres xcen 0 ycen 0 zcen 0
        # select resname tip3 end
        image.setup_residue(0.0, 0.0, 0.0, "TIP3")
        # CHARMM scripting: image byres xcen 0 ycen 0 zcen 0
        # select resname ion_type end
        image.setup_residue(0.0, 0.0, 0.0, "DCM")

    def minimize(self, dict=None, type="abnr"):
        """Minimize the structure"""
        if type == "abnr":
            if dict is None:
                _dict = ABNR_DICT
            else:
                _dict = dict
            minimize.run_abnr(**_dict)

    def write_res_dcd(self, resname):
        """Write the restart and dcd files

        :param resname: str of the key for saving the files
                        KEY.res and KEY.dcd will be created in
                        the output_path
        :return: None
        """
        res_file = pycharmm.CharmmFile(
            file_name=f"{self.output_path}/{resname}.res",
            file_unit=2,
            formatted=True,
            read_only=False,
        )
        dcd_file = pycharmm.CharmmFile(
            file_name=f"{self.output_path}/{resname}.dcd",
            file_unit=1,
            formatted=False,
            read_only=False,
        )
        return res_file, dcd_file

    def read_restart(self, resname):
        """Read the restart file into UNIT 3"""
        print(f"Reading restart file {self.output_path}/{resname}.res")
        restart_file = pycharmm.CharmmFile(
            file_name=f"{self.output_path}/{resname}.res",
            file_unit=3,
            formatted=True,
            read_only=True,
        )
        return restart_file

    def write_info(self, key, note=""):
        """Write the pdb and psf files for the structure"""
        write.coor_pdb(f"{self.output_path}/{key}.pdb", title=f"{key}: {note}")
        write.psf_card(f"{self.output_path}/{key}.psf", title=f"{key}: {note}")

    def dynamics(
        self,
        resname="heat",
        dynatype="heat",
        restart=False,
        pmass=None,
        nstep=10000,
        timestep=0.0005,
    ):
        # prepare the dynamics script
        res_file, dcd_file = self.write_res_dcd(resname)

        if restart:
            restart_file = self.read_restart(restart)
        else:
            restart_file = None

        dyna_dict = get_dynamics_dict(
            timestep,
            res_file,
            dcd_file,
            dynatype=dynatype,
            restart=restart_file,
            str_file=restart_file,
            pmass=pmass,
            nstep=nstep,
        )

        # load the dynamics script
        dyn_ = pycharmm.DynamicsScript(**dyna_dict)
        # run
        dyn_.run()
        # close files
        res_file.close()
        dcd_file.close()
        if restart:
            restart_file.close()

    def routine(self):
        """Basic routine for doing minimization, heating, equilibration
        and production dynamics

        :return: None
        """
        #  setup
        pmass = get_pmass(self.pdb_name)
        self.setup()
        self.imageinit()

        # dcm setup if needed
        if self.dcm_file_path is not None:
            print("DCM setup")
            self.dcm_setup()

        # minimization
        self.minimize()
        self.write_info("dcm_box", note="dcm_box")
        self.dynamics(dynatype="heat", nstep=10000, restart=False)
        self.write_info("dcm_box_heat", note="dcm_box_heat")

        #  equilibration
        self.dynamics(
            dynatype="equil", nstep=10000, resname="equil", restart="heat", pmass=pmass
        )
        self.write_info("dcm_box_equil", note="dcm_box_equil")

        # production dynamics
        self.dynamics(
            dynatype="prod", nstep=10000, resname="prod", restart="equil", pmass=pmass
        )
        self.write_info("dcm_box_prod", note="dcm_box_prod")


if __name__ == "__main__":
    pcr = PyCHARMM_Runner(DCM_TEST)
    print(pcr)
    pcr.routine()
