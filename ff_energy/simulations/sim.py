from ff_energy.simulations.templater import SimTemplate, input_files
from ff_energy.ffe.ffe_utils import makeDir
from pathlib import Path
from shutil import copyfile


class Simulation:
    """A class to set up input for a simulation
    in CHARMM
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        #
        self.job_path = Path(kwargs["job_path"])
        self.basepathname = Path(kwargs["BASEPATHNAME"])
        self.T = kwargs["TEMP"]
        self.updated = False
        # update the paths for a given temp.
        self.update_paths()

        # make the simulation template
        self.sim = SimTemplate(self.kwargs)

    def make(self):
        # make the job path if it doesn't exist
        makeDir(self.job_path)
        # copy the files to the job path
        self.copy_files()
        # make the input file
        self.make_input()
        # make the submit file
        self.make_submit()

    def set_T(self, T):
        self.T = T
        self.kwargs["TEMP"] = float(T)
        self.TEMP = float(T)
        self.update_paths()


    def update_paths(self):
        if not self.updated:
            self.kwargs["job_path"] = self.job_path / f"k{self.T}"
            self.kwargs["BASEPATHNAME"] = self.basepathname / f"k{self.T}"
        else:
            self.kwargs["job_path"] = self.job_path.parents[0] / f"k{self.T}"
            self.kwargs["BASEPATHNAME"] = self.basepathname.parents[0] / f"k{self.T}"


        self.job_path = self.kwargs["job_path"]
        self.basepathname = self.kwargs["BASEPATHNAME"]

        self.updated = True

        print(self.kwargs["job_path"])
        print(self.kwargs["BASEPATHNAME"])

    def make_input(self):
        inp = self.sim.get_sim()
        with open(self.job_path / "dynamics.inp", "w") as f:
            f.write(inp)

    def make_submit(self):
        sub = self.sim.get_submit()
        with open(self.job_path / "job.sh", "w") as f:
            f.write(sub)

    def copy_files(self):

        self.kwargs["extrafiles"].append(self.kwargs["PSFNAME"])
        self.kwargs["extrafiles"].append(self.kwargs["CRDNAME"])

        # get the input file names
        input_file_names = [str(_.name) for _ in list(input_files.glob("*"))]
        # copy extra files
        for fi in self.kwargs["extrafiles"]:
            print(fi)
            if fi in input_file_names:
                copyfile(input_files / fi,
                         self.job_path / fi)
            else:
                raise FileNotFoundError(f"{fi} not found in {input_files}")
