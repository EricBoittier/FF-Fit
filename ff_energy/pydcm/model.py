import json
from pathlib import Path, PosixPath

from ff_energy.ffe.constants import FFEPATH


class DCM:
    """
    DCM model class
    """

    def __init__(self, config=None):
        pass


if __name__ == "__main__":
    """
    $$$                                                             $$$
    This script is used to generate the json files for the DCM models
    $$$                                                            $$$
    Water
    """
    water_dict = {}
    water_mdcm_path = Path(__file__).parents[0] / "sources" / "water"
    water_cubes_path = FFEPATH / "cubes" / "water"
    print(water_cubes_path)
    water_dict["scan_fesp"] = [
        str(_) for _ in list(water_cubes_path.glob("*/*_esp.cube"))
    ]
    print(water_dict["scan_fesp"])
    water_dict["scan_fdns"] = [
        str(_) for _ in list(water_cubes_path.glob("*/*_dens.cube"))
    ]
    water_dict["mdcm_cxyz"] = str(water_mdcm_path / "refined.xyz")
    water_dict["mdcm_clcl"] = str(water_mdcm_path / "pbe0_dz.mdcm")
    json.dump(water_dict, open("tests/water.json", "w"))

    """
    Methanol
    """
    methanol_dict = {}
    methanol_mdcm_path = Path(__file__).parents[0] / "sources" / "methanol"
    methanol_cubes_path = FFEPATH / "cubes" / "methanol"
    methanol_dict["scan_fesp"] = [
        str(_) for _ in list(methanol_cubes_path.glob("*/*_esp.cube"))
    ]
    methanol_dict["scan_fdns"] = [
        str(_) for _ in list(methanol_cubes_path.glob("*/*_dens.cube"))
    ]
    methanol_dict["mdcm_cxyz"] = str(methanol_mdcm_path / "refined.xyz")
    methanol_dict["mdcm_clcl"] = str(methanol_mdcm_path / "meoh_pbe0dz.mdcm")
    json.dump(methanol_dict, open("tests/methanol.json", "w"))

    """
    Methanol with Permutations
    """
    methanol_dict = {}
    methanol_mdcm_path = Path(__file__).parents[0] / "sources" / "methanol"
    methanol_cubes_path = FFEPATH / "cubes" / "methanol_perm"
    methanol_dict["scan_fesp"] = [
        str(_) for _ in list(methanol_cubes_path.glob("*/*_esp.cube"))
    ]
    methanol_dict["scan_fdns"] = [
        str(_) for _ in list(methanol_cubes_path.glob("*/*_dens.cube"))
    ]
    methanol_dict["mdcm_cxyz"] = str(methanol_mdcm_path / "refined.xyz")
    methanol_dict["mdcm_clcl"] = str(methanol_mdcm_path / "meoh_pbe0dz.mdcm")
    json.dump(methanol_dict, open("tests/methanol_perm.json", "w"))

    """
    Dichloromethane
    """
    dcm_dict = {}
    dcm_mdcm_path = FFEPATH / "ff_energy/pydcm" / "sources" / "dcm"
    dcm_cubes_path = FFEPATH / "cubes" / "dcm"
    dcm_dict["scan_fesp"] = [str(_) for _ in list(dcm_cubes_path.glob("*/*_esp.cube"))]
    dcm_dict["scan_fdns"] = [str(_) for _ in list(dcm_cubes_path.glob("*/*_dens.cube"))]
    dcm_dict["mdcm_cxyz"] = str(dcm_mdcm_path / "dcm8.xyz")
    dcm_dict["mdcm_clcl"] = str(dcm_mdcm_path / "dcm.mdcm")
    json.dump(dcm_dict, open("tests/dcm.json", "w"))
