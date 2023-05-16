import pickle
from pathlib import Path
import re

H2KCALMOL = 627.503

#  dynamic path to pickle folder
PKL_PATH = Path(__file__).parents[2] / "pickles"


def get_structures(system_name):
    pickle_exists = Path(PKL_PATH / f"structures/{system_name}.pkl").exists()
    if pickle_exists:
        print("Strcuture,PDB already already exists, loading structure from pickle")
        structures, pdbs = next(read_from_pickle(PKL_PATH /
                                                 f"structures/{system_name}.pkl"))
        return structures, pdbs
    else:
        raise FileNotFoundError(f"Pickle {system_name} does not exist")
        # return False, False


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
