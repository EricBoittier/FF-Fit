import pickle
from pathlib import Path
import pandas as pd
from ff_energy.data import Data



def pickle_output(output,name="dists"):
    pickle_path = Path(f'pickles/{name}.pkl')
    pickle_path.parents[0].mkdir(parents=True, exist_ok=True)
    with open(pickle_path, 'wb') as handle:
        pickle.dump(output, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

def read_from_pickle(path):
    with open(path, 'rb') as file:
        try:
            while True:
                yield pickle.load(file)
        except EOFError:
            pass
