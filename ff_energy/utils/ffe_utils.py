import logging
import pickle
from pathlib import Path
import re
import numpy as np
import codecs, json

from ff_energy.ffe.jobmaker import get_structures_pdbs, JobMaker
from ff_energy.ffe.constants import atom_types, PDB_PATH
from ff_energy.logs.logging import logger

sysname_to_res = {
    "water_cluster": "LIG",
    "dcm": "DCM",
    "ions_ext": "LIG",
    "dcmdimerscan": "DCM",
}


def MakeJob(name, config_maker, _atom_types=None, system_name=None, RES=None):
    """ """
    if _atom_types is None:
        _atom_types = atom_types

    pickle_exists = get_structures(system_name,
                                   pdbpath=config_maker.pdbs)
    if pickle_exists[0]:
        structures, pdbs = pickle_exists
    else:
        logger.warning(f"pickle ({system_name}) does not exist")
        pdbpath = Path(config_maker.pdbs)
        pdbpath = PDB_PATH / pdbpath if not pdbpath.is_absolute() else pdbpath
        structures, pdbs = get_structures_pdbs(
            pdbpath,
            atom_types=_atom_types, system_name=system_name
        )
        pickle_output((structures, pdbs), name=system_name)

    return JobMaker(name, pdbs, structures, config_maker.make().__dict__, RES=RES)


def charmm_jobs(CMS):
    jobmakers = []
    for cms in CMS:
        logging.info(cms.elec)
        jm = MakeJob(
            f"{cms.system_name}/{cms.theory_name}_{cms.elec}",
            cms,
            _atom_types=cms.atom_types,
            system_name=cms.system_name,
            RES=sysname_to_res[cms.system_name],
        )
        HOMEDIR = "/home/boittier/homeb/"
        f"/home/boittier/pcbach/{cms.system_name}/{cms.theory_name}"
        jm.make_charmm(HOMEDIR)
        jobmakers.append(jm)
    return jobmakers


H2KCALMOL = 627.503

#  dynamic path to pickle folder
PKL_PATH = Path(__file__).parents[2] / "pickles"


def get_structures(system_name, pdbpath=None):
    """
    Get structures and pdbs from pickle file, if no path is given,
    load from the pickles folder
    :param system_name:
    :param pdbpath:
    :return:
    """
    pickle_exists = Path(PKL_PATH / f"structures/{system_name}.pkl").exists()
    if pickle_exists:
        print("Structure/PDB already already exists, loading from pickle")
        structures, pdbs = next(
            read_from_pickle(PKL_PATH / f"structures/{system_name}.pkl")
        )
    elif pdbpath is None:
        raise ValueError("pdbpath must be specified")
    else:
        print("Structure/PDB does not exist, creating pickle",
              f"structures/{system_name}.pkl",
              "from", pdbpath)
        structures, pdbs = get_structures_pdbs(Path(pdbpath), system_name=system_name)
        pickle_output((structures, pdbs), name=f"structures/{system_name}")
    print("Structures/PDBs loaded", len(structures), len(pdbs))
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
    :param g: string
    :return: integer
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


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def write_json(data, path):
    keys = data.keys()
    fn = "_".join(keys) + ".json"
    path = Path(path) / fn
    with codecs.open(path, "w", encoding="utf-8") as f:
        json.dump(
            data, f, separators=(",", ":"), sort_keys=True, indent=4, cls=NumpyEncoder
        )


def json_load_np(path: str) -> dict:
    json_load = json.loads(path)
    np_dict = {}  # dictionary of numpy objects
    for key in json_load.keys():
        np_dict[key] = np.array(json_load[key])
    return np_dict
