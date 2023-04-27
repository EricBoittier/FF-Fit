import pickle
from pathlib import Path

#  dynamic path to pickle folder
PKL_PATH = Path(__file__).parents[1] / "pickles"


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
