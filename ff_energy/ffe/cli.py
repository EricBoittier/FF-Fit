from ff_energy.ffe.constants import CONFIG_PATH, CLUSTER_DRIVE, clusterBACH, \
    clusterNCCR, clusterBEETHOVEN
from ff_energy.ffe.utils import MakeJob, charmm_jobs
from ff_energy.ffe.utils import pickle_output
from ff_energy.ffe.utils import PKL_PATH
from ff_energy.ffe.configmaker import ConfigMaker, system_names, THEORY
from ff_energy.ffe.config import Config
from ff_energy.ffe.data import Data
from pathlib import Path
from ff_energy.ffe.slurm import SlurmJobHandler
import sys



def load_config_maker(theory, system, elec):
    cm = ConfigMaker(theory, system, elec)
    return [cm]


def load_config_from_input(filenames) -> []:
    CMS = []
    filenames = filenames.split()
    for filename in filenames:
        conf = Config()
        conf.read_config(CONFIG_PATH + filename)
        CMS.append(conf)
    return CMS


def load_all_theory_and_elec():
    CMS = []
    for system in system_names:
        for theory in THEORY.keys():
            for elec in ["pc", "mdcm"]:
                cm = ConfigMaker(theory, system, elec)
                CMS.append(cm)
    return CMS


def load_all_theory():
    CMS = []
    for system in system_names:
        for theory in THEORY.keys():
            cm = ConfigMaker(theory, system, "pc")
            CMS.append(cm)

    return CMS





def submit_jobs(jobs, max_jobs=120, Check=True, cluster=clusterBACH):
    shj = SlurmJobHandler(max_jobs=max_jobs, cluster=cluster)
    print("Running jobs.py: ", shj.get_running_jobs())
    for j in jobs:
        shj.add_job(j)

    print("Jobs: ", len(shj.jobs))
    print(len(shj.jobs))
    shj.shuffle_jobs()
    shj.submit_jobs(Check=Check)


def coloumb_submit(cluster, jobmakers, max_jobs=120, Check=True):
    jobs = []
    for jm in jobmakers:
        DRIVE = CLUSTER_DRIVE[cluster[1]]
        for js in jm.get_coloumb_jobs(DRIVE):
            jobs.append(js)

    submit_jobs(jobs, max_jobs=max_jobs, Check=Check, cluster=cluster)


def charmm_submit(cluster, jobmakers, max_jobs=120, Check=True):
    jobs = []
    for jm in jobmakers:
        DRIVE = CLUSTER_DRIVE[cluster[1]]
        for js in jm.get_charmm_jobs(DRIVE):
            jobs.append(js)

    submit_jobs(jobs, max_jobs=max_jobs, Check=Check, cluster=cluster)


def molpro_submit_big(cluster, jobmakers, max_jobs=120, Check=True):
    jobs = []
    for jm in jobmakers:
        DRIVE = CLUSTER_DRIVE[cluster[1]]

        for js in jm.get_cluster_jobs(DRIVE):
            jobs.append(js)

    submit_jobs(jobs, max_jobs=max_jobs, Check=Check, cluster=cluster)


def molpro_submit_small(cluster, jobmakers, max_jobs=120, Check=True):
    jobs = []
    for jm in jobmakers:
        DRIVE = CLUSTER_DRIVE[cluster[1]]
        for js in jm.get_monomer_jobs(DRIVE):
            jobs.append(js)

        for js in jm.get_pairs_jobs(DRIVE):
            jobs.append(js)
    submit_jobs(jobs, max_jobs=max_jobs, Check=Check, cluster=cluster)


def molpro_jobs_big(CMS, DRY):
    jobmakers = []
    for cms in CMS:
        print(cms)
        jm = MakeJob(
            f"{cms.system_name}/{cms.theory_name}",
            cms,
            _atom_types=cms.atom_types,
            system_name=cms.system_name,
        )
        "/home/boittier/homeb/"
        PCBACH = "/home/boittier/pcbach/"  # {cms.system_name}/{cms.theory_name}"
        if not DRY:
            jm.make_molpro(PCBACH)
        jobmakers.append(jm)
    return jobmakers


def molpro_jobs_small(CMS, DRY):
    jobmakers = []
    for cms in CMS:
        print(cms)
        jm = MakeJob(
            f"{cms.system_name}/{cms.theory_name}",
            cms,
            _atom_types=cms.atom_types,
            system_name=cms.system_name,
        )
        "/home/boittier/pcnccr/"
        PCBACH = "/home/boittier/pcnccr/"  # {cms.system_name}/{cms.theory_name}"
        if not DRY:
            jm.make_molpro(PCBACH)
        jobmakers.append(jm)
    return jobmakers


def data_jobs(CMS, molpro_small_path):
    jobmakers = []
    cms = None
    jm = None
    for cms in CMS:
        print(cms)
        jm = MakeJob(
            f"{cms.system_name}/{cms.theory_name}_{cms.elec}",
            cms,
            _atom_types=cms.atom_types,
            system_name=cms.system_name,
        )
        HOMEDIR = "/home/boittier/homeb/"
        PCBACH = f"/home/boittier/pcbach/{cms.system_name}/{cms.theory_name}"
        PCNCCR = f"/home/boittier/pcnccr/{cms.system_name}/{cms.theory_name}"
        COLOUMB = f"/home/boittier/homeb/{cms.system_name}/{cms.theory_name}"
        CHM = f"/home/boittier/homeb/{cms.system_name}/{cms.theory_name}_{cms.elec}"

        PAIRS = PCNCCR
        MONOMERS = PCNCCR
        if molpro_small_path is not None:
            PAIRS = molpro_small_path + f"/{cms.system_name}/{cms.theory_name}"
            MONOMERS = molpro_small_path + f"/{cms.system_name}/{cms.theory_name}"
        print("pairs path:", PAIRS)
        print("monomers path:", MONOMERS)
        jm.gather_data(
            HOMEDIR,
            MONOMERS,  # monomers
            PCBACH,  # cluster
            PAIRS,  # pairs
            COLOUMB,  # coulomb
            CHM, # charmm
        )
        jobmakers.append(jm)

    #  convert data to data object
    pp = Path(PKL_PATH / f"{cms.system_name}/{cms.theory_name}/"
                         f"{jm.kwargs['c_files'][0]}")
    print("Saving data to: ", pp)
    data = Data(pp)
    pickle_output(data, PKL_PATH / f"{cms.system_name}_{cms.theory_name}_{cms.elec}")
    return jobmakers


def esp_view_jobs(CMS):
    for cms in CMS:
        print(cms)
        jm = MakeJob(
            f"{cms.system_name}/{cms.theory_name}_{cms.elec}",
            cms,
            _atom_types=cms.atom_types,
            system_name=cms.system_name,
        )
        HOMEDIR = "/home/boittier/homepcb/"
        f"/home/boittier/pcbach/{cms.system_name}/{cms.theory_name}"
        f"/home/boittier/homeb/{cms.system_name}/{cms.theory_name}"
        CHM = f"/home/boittier/homeb/{cms.system_name}/{cms.theory_name}_{cms.elec}"
        jm.esp_view(HOMEDIR, CHM)


def coloumb_jobs(CMS, DRY, cluster=None):
    jobmakers = []
    for cms in CMS:
        print(cms)
        jm = MakeJob(
            f"{cms.system_name}/{cms.theory_name}",
            cms,
            _atom_types=cms.atom_types,
            system_name=cms.system_name,
        )
        HOMEDIR = "/home/boittier/homeb/"
        PCBACH = f"/home/boittier/pcbach/{cms.system_name}/{cms.theory_name}"
        if cluster is not None:
            PCBACH = f"{cluster}/{cms.system_name}/{cms.theory_name}"
        if not DRY:
            jm.make_coloumb(HOMEDIR, PCBACH + "/{}/monomers")
        jobmakers.append(jm)
    return jobmakers


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="ProgramName",
        description="What the program does",
        epilog="Text at the bottom of help",
    )
    print("----")

    # parser.add_argument('filename')           # positional argument
    parser.add_argument(
        "-d", "--data", required=False, default=False, action="store_true"
    )  # option that takes a value
    parser.add_argument(
        "-a", "--all", required=False, default=False, action="store_true"
    )
    parser.add_argument(
        "-at", "--all_theory", required=False, default=False, action="store_true"
    )
    parser.add_argument(
        "-c", "--cluster", required=False, default=False, action="store_true"
    )
    parser.add_argument(
        "-x", "--config", required=False, default=False, action="store_true"
    )
    parser.add_argument("-t", "--theory", required=False, default=None)
    parser.add_argument("-m", "--model", required=False, default=None)
    parser.add_argument("-e", "--elec", required=False, default=None)
    parser.add_argument(
        "-s", "--submit", required=False, default=False, action="store_true"
    )
    parser.add_argument(
        "-cj", "--coloumb", required=False, default=False, action="store_true"
    )
    parser.add_argument(
        "-chmj", "--chm", required=False, default=False, action="store_true"
    )
    parser.add_argument(
        "-mj", "--molpro", required=False, default=False, action="store_true"
    )
    parser.add_argument(
        "-mjs", "--molpro_small", required=False, default=False, action="store_true"
    )
    parser.add_argument("-msp", "--molpro_small_path", required=False, default=None)

    parser.add_argument(
        "-dry", "--dry", required=False, default=False, action="store_true"
    )
    parser.add_argument("-v", "--verbose", action="store_true")  # on/off flag

    CMS = None
    args = parser.parse_args()

    if args.verbose:
        print(args)

    if args.all:
        if args.verbose:
            print("Loading all data")
        CMS = load_all_theory_and_elec()
    elif args.all_theory:
        if args.verbose:
            print("Loading all theory")
        CMS = load_all_theory()
    elif args.config:
        if args.verbose:
            print("Making Configs: ", args.config)
        CMS = load_config_from_input(args.config)
    else:
        print(args.theory, args.model, args.elec)
        if args.theory and args.model and args.elec:
            CMS = load_config_maker(args.theory, args.model, args.elec)
        else:
            print("Missing one of args.theory and args.model and args.elec")
            print(parser.print_help())
            sys.exit(1)

    if CMS is not None:
        for c in CMS:
            c.write_config()

        if args.molpro:
            if args.verbose:
                print("Making big Molpro Jobs")
            jobmakers = molpro_jobs_big(CMS, args.dry)
            if args.submit:
                if args.verbose:
                    print("Submitting Molpro Jobs")
                molpro_submit_big(clusterBACH, jobmakers, max_jobs=400, Check=True)

        if args.molpro_small:
            if args.verbose:
                print("Making small Molpro Jobs")
            jobmakers = molpro_jobs_small(CMS, args.dry)
            if args.submit:
                if args.verbose:
                    print("Submitting Molpro Jobs")
                molpro_submit_small(clusterNCCR, jobmakers, max_jobs=400, Check=True)

        if args.chm:
            if args.verbose:
                print("Making CHARMM Jobs")
            jobmakers = charmm_jobs(CMS)
            if args.submit:
                if args.verbose:
                    print("Submitting CHM Jobs")
                charmm_submit(clusterBEETHOVEN, jobmakers, max_jobs=120, Check=False)

        if args.coloumb:
            if args.verbose:
                print("Making Coloumb Jobs")
            jobmakers = coloumb_jobs(CMS, args.dry, cluster=args.molpro_small_path)
            if args.submit:
                if args.verbose:
                    print("Submitting Coloumb Jobs")
                coloumb_submit(clusterBEETHOVEN, jobmakers, max_jobs=120, Check=True)

        if args.data:
            if args.verbose:
                print("Gathering Data")
            print("MSP:", args.molpro_small_path)
            jobmakers = data_jobs(CMS, args.molpro_small_path)

    else:
        print("No Jobs Found...")
