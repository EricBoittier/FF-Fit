from templater import SimTemplate, template_files, input_files
from ff_energy.ffe.utils import makeDir
from pathlib import Path
from shutil import copyfile


class Simulation:
    """A class to set up input for a simulation
    in CHARMM
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.sim = SimTemplate(kwargs)
        self.job_path = Path(kwargs["job_path"])
        # make the job path if it doesn't exist
        makeDir(self.job_path)
        # copy the files to the job path
        self.copy_files()
        # make the input file
        self.make_input()
        # make the submit file
        self.make_submit()

    def make_input(self):
        inp = self.sim.get_sim()
        with open(self.job_path / "dynamics.inp", "w") as f:
            f.write(inp)

    def make_submit(self):
        sub = self.sim.get_submit()
        with open(self.job_path / "job.sh", "w") as f:
            f.write(sub)

    def copy_files(self):
        # copy parameter file
        if self.kwargs["PAR"] in input_files.glob("*"):
            copyfile(template_files / self.kwargs["PAR"],
                     self.job_path / self.kwargs["PAR"])
        # copy topology file
        if self.kwargs["TOP"] in input_files.glob("*"):
            copyfile(template_files / self.kwargs["TOP"],
                     self.job_path / self.kwargs["TOP"])
        # copy extra files
        for fi in self.kwargs["extrafiles"]:
            if fi in input_files.glob("*"):
                copyfile(template_files / fi,
                         self.job_path / fi)
