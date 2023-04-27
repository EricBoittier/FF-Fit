import json

kwargs = {
    "m_nproc": 8,
    "m_memory": 150,
    "m_queue": "short",
    "m_basis": "avdz",
    "m_method": "hf",
    "chmpath": "/home/boittier/dev-release-dcm/build/cmake/charmm",
    "modules": "module load cmake/cmake-3.23.0-gcc-11.2.0-openmpi-4.1.3",
    "c_files": ["pbe0_dz.pc"],
    "c_dcm_command": "open unit 11 card read name pbe0_dz.pc \nDCM IUDCM 11 TSHIFT XYZ 15",
}


def DCM_STATES(x):
    return {
        "c_files": [x],
        "c_dcm_command": f"open unit 11 card read name {x} \nDCM IUDCM 11 TSHIFT XYZ 15",
    }


def kMDCM_STATES(x):
    return {
        "c_files": x,
        "c_dcm_command": f"open unit 11 card read name {x[0]} \n"
        f"open unit 12 card read name {x[1].split('/')[-1]} \n"
        f"DCM KERN 12 IUDCM 11 TSHIFT XYZ 15",
    }


class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        if "m_nproc" not in self.__dict__:
            self.m_nproc = 4
        if "m_memory" not in self.__dict__:
            self.m_memory = 150
        if "m_queue" not in self.__dict__:
            self.m_queue = "short"
        if "m_basis" not in self.__dict__:
            self.m_basis = "avdz"
        if "m_method" not in self.__dict__:
            self.m_method = "hf"
        if "chmpath" not in self.__dict__:
            self.chmpath = "/home/boittier/dev-release-dcm/build/cmake/charmm"
        if "modules" not in self.__dict__:
            self.modules = "module load cmake/cmake-3.23.0-gcc-11.2.0-openmpi-4.1.3"
        if "c_files" not in self.__dict__:
            self.c_files = ["pbe0_dz.pc"]
        if "c_dcm_command" not in self.__dict__:
            self.c_dcm_command = (
                "open unit 11 card read name pbe0_dz.pc \nDCM IUDCM 11 TSHIFT XYZ 15"
            )

    def kwargs(self):
        return self.__dict__

    def __str__(self):
        return "".join([f"{k}: {v}\n" for k, v in self.__dict__.items()])

    def __repr__(self):
        return self.__str__()

    def set_dcm(self, dcm):
        dcm = DCM_STATES(dcm)
        self.c_files = dcm["c_files"]
        self.c_dcm_command = dcm["c_dcm_command"]

    def set_kmdcm(self, kmdcm):
        kmdcm = kMDCM_STATES(kmdcm)
        self.c_files = kmdcm["c_files"]
        self.c_dcm_command = kmdcm["c_dcm_command"]

    def set_theory(self, theory):
        self.m_method = theory[0]
        self.m_basis = theory[1]

    def write_config(self, filename):
        # filename = f"{self.}"
        with open(filename, "w") as f:
            f.write(json.dumps(self.__dict__))

    def read_config(self, filename):
        with open(filename, "r") as f:
            self.__dict__.update(json.loads(f.read()))


# print(Config(**kwargs))
# DEFAULT_CONFIG = Config(**kwargs)
# print(DEFAULT_CONFIG)
