import pickle
from pathlib import Path
import re

from ff_energy.ffe.jobmaker import get_structures_pdbs, JobMaker
from ff_energy.ffe.constants import atom_types


def MakeJob(name,
            config_maker,
            _atom_types=None,
            system_name=None
            ):
    if _atom_types is None:
        _atom_types = atom_types

    pickle_exists = get_structures(system_name)
    if pickle_exists[0]:
        structures, pdbs = pickle_exists
    else:
        print(f"pickle ({system_name}) does not exist")
        structures, pdbs = get_structures_pdbs(
            Path(config_maker.pdbs), atom_types=_atom_types, system_name=system_name
        )
        pickle_output((structures, pdbs), name=system_name)

    return JobMaker(name, pdbs, structures, config_maker.make().__dict__)


def charmm_jobs(CMS):
    jobmakers = []
    for cms in CMS:
        print(cms.elec)
        jm = MakeJob(
            f"{cms.system_name}/{cms.theory_name}_{cms.elec}",
            cms,
            _atom_types=cms.atom_types,
            system_name=cms.system_name,
        )
        HOMEDIR = "/home/boittier/homeb/"
        f"/home/boittier/pcbach/{cms.system_name}/{cms.theory_name}"
        jm.make_charmm(HOMEDIR)
        jobmakers.append(jm)
    return jobmakers


H2KCALMOL = 627.503

#  dynamic path to pickle folder
PKL_PATH = Path(__file__).parents[2] / "pickles"


def get_structures(system_name, pdbpath="pdbs/water_tests"):
    pickle_exists = Path(PKL_PATH / f"structures/{system_name}.pkl").exists()
    if pickle_exists:
        print("Strcuture/PDB already already exists, loading from pickle")
        structures, pdbs = next(read_from_pickle(PKL_PATH /
                                                 f"structures/{system_name}.pkl"))
    else:
        structures, pdbs = get_structures_pdbs(Path(pdbpath), system_name=system_name)
        pickle_output((structures, pdbs), name=f"structures/{system_name}")

    return structures, pdbs


def makeDir(path: Path):
    """
    Make a directory if it doesn't exist
    :param path:
    :return:
    """
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def str2int(g):
    """
    Convert a string to an integer
    :param g:
    :return:
    """
    return int(re.sub("[^0-9]", "", g))


def pickle_output(output, name="dists"):
    pickle_path = Path(PKL_PATH / f"{name}.pkl")
    pickle_path.parents[0].mkdir(parents=True, exist_ok=True)
    with open(pickle_path, "wb") as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_from_pickle(path):
    """
    Read from pickle file, if no path is given, load from the pickles folder
    :param path:
    :return:
    """
    #  check if path is a string or a path object
    if isinstance(path, str):
        path = Path(path)
        if not path.exists():
            path = Path(PKL_PATH / path)
    elif isinstance(path, Path):
        if not path.exists():
            path = Path(PKL_PATH / path)

    with open(path, "rb") as file:
        try:
            while True:
                yield pickle.load(file)
        except EOFError:
            print("EOFError when loading {}".format(path))
