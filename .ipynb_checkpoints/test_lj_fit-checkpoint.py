# from ff_energy.structure import
from ff_energy.potential import FF, LJ
from ff_energy.data import Data
import pickle
from pathlib import Path

def p_data(pk):
    d = Data(pk)
    return d

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

# CMS = load_config_maker("pbe0dz", "water_cluster", "mdcm")
# jobs.py = charmm_jobs(CMS)
# dists = {_.name.split(".")[0]: _.distances for _ in jobs.py[0].structures}
# structures = [_ for _ in jobs.py[0].structures]
# pickle_output(structures, name="structures")
# pickle_output(dists)

dists = next(read_from_pickle("pickles/structures/dists.pkl"))
structures = next(read_from_pickle("pickles/structures/structures.pkl"))
# exit()
# dists = next(dists)
# print(dists.keys())
# exit()
# keys = [_.name for _ in jobs.py[0].structures]
# print(keys[:10])
pk = "pickles/water_cluster/pbe0_dz.mdcm"
pbe0_dz_mdcm = p_data(pk)
# pbe0_dz_mdcm.data
s = structures[0]

sig_bound = (0.5, 4.0)
ep_bound = (0.00001, 1.0)
LJ_bound = [(sig_bound), (sig_bound), (ep_bound), (ep_bound)]

ff = FF(pbe0_dz_mdcm.data,dists,LJ,LJ_bound,s)
d = pbe0_dz_mdcm.data.sample(100)

# _ = ff.LJ_performace(d)
# plot_intE(_)
# plt.show()

# exit()
args = [1.7682,0.2245,-0.1521,-0.0460]

# print("RMSE:", np.sqrt(ff.get_loss(args)))
# print(ff.mse_df)

# print(LJ_bound)

# print(len(ff.all_sig))
# print(type(ff.all_sig))
# print(len(ff.all_sig[0]))
# print(type(ff.all_sig[0]))

# for i in range(1000):
    # rmse = np.sqrt(ff.get_loss(args))
    # print("RMSE:", rmse)

# exit()
#
ff.fit_repeat(10, bounds=LJ_bound)
pickle_output(ff, "ff/pbe0_dz_mdcm")
# plot_intE(ff.mse_df)
# plt.show()
