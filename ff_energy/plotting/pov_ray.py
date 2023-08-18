import numpy as np
from pathlib import Path
import os
import shutil

from ase import io
from ase.io.pov import get_bondpairs
from ase.data import covalent_radii


def render_povray(atoms, pov_name,
                  rotation='0x, 0y, 0z',
                  radius_scale=0.40):
    print("Rendering POV-Ray image...")
    print("path: ", pov_name)

    path = Path(pov_name)
    pov_name = path.name
    base = path.parent

    color_dict = {
        'Cl': [102, 227, 115],
        'C': [61, 61, 64],
        'O': [240, 10, 10],
        'H': [232, 206, 202],
        'X': [200, 200, 200]}

    radius_scale = 0.40

    radius_list = []
    for atomic_number in atoms.get_atomic_numbers():
        radius_list.append(radius_scale * covalent_radii[atomic_number])
    colors = np.array([color_dict[atom.symbol] for atom in atoms]) / 255

    bondpairs = get_bondpairs(atoms, radius=1.1)
    good_bonds = []
    for _ in bondpairs:
        #  remove the Cl-Cl bonds
        if not (atoms[_[0]].symbol == "Cl" and atoms[_[1]].symbol == "Cl"):
            good_bonds.append(_)

    kwargs = {  # For povray files only
        'transparent': True,  # Transparent background
        'canvas_width': 1028,  # Width of canvas in pixels
        'canvas_height': None,  # None,  # Height of canvas in pixels
        'camera_dist': 50.0,  # Distance from camera to front atom,
        'camera_type': 'orthographic angle 0',  # 'perspective angle 20'
        'depth_cueing': False,
        'colors': colors,
        'bondatoms': good_bonds,
        "textures": ["jmol"] * len(atoms),
    }

    generic_projection_settings = {
        'rotation': rotation,
        'radii': radius_list,
    }

    povobj = io.write(
        pov_name,
        atoms,
        **generic_projection_settings,
        povray_settings=kwargs)

    povobj.render()
    png_name = pov_name.replace(".pov", ".png")
    shutil.move(png_name, base / png_name)
