from pycharmm import CharmmFile


def get_dynamics_dict(timestep=0.0005,
                      res_file: CharmmFile = None,
                      dcd_file: CharmmFile = None,
                      nstep=None,):
    """

    :param timestep:
    :param res_file:
    :param dcd_file:
    :return:
    """
    if res_file is None or dcd_file is None:
        raise ValueError("res_file and dcd_file must be specified")

    # Run some dynamics
    dynamics_dict = {
        'leap': False,
        'verlet': True,
        'cpt': False,
        'new': False,
        'langevin': False,
        'timestep': timestep,
        'start': True,
        'nstep': 5. * 1. / timestep,
        'nsavc': 0.01 * 1. / timestep,
        'nsavv': 0,
        'inbfrq': -1,
        'ihbfrq': 50,
        'ilbfrq': 50,
        'imgfrq': 50,
        'ixtfrq': 1000,
        'iunrea': -1,
        'iunwri': res_file.file_unit,
        'iuncrd': dcd_file.file_unit,
        'nsavl': 0,  # frequency for saving lambda values in lamda-dynamics
        'iunldm': -1,
        'ilap': -1,
        'ilaf': -1,
        'nprint': 100,  # Frequency to write to output
        'iprfrq': 500,  # Frequency to calculate averages
        'isvfrq': 1000,  # Frequency to save restart file
        'ntrfrq': 1000,
        'ihtfrq': 200,
        'ieqfrq': 1000,
        'firstt': 100,
        'finalt': 300,
        'tbath': 300,
        'iasors': 0,
        'iasvel': 1,
        'ichecw': 0,
        'iscale': 0,  # scale velocities on a restart
        'scale': 1,  # scaling factor for velocity scaling
        'echeck': -1}

    if nstep is not None:
        dynamics_dict['nstep'] = nstep

    print("Pychramm dynamics dict:")
    for k,v in dynamics_dict.items():
        print(k, v)

    return dynamics_dict
