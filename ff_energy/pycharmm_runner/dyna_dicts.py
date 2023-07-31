from pycharmm import CharmmFile


def get_dynamics_dict(timestep=0.0005,
                      res_file: CharmmFile = None,
                      dcd_file: CharmmFile = None,
                      str_file: CharmmFile = None,
                      dynatype="heat",
                      restart=False,
                      pmass=None,
                      nstep=None, ):
    """
    Returns a dictionary of parameters for the dynamics script
    """
    if res_file is None or dcd_file is None:
        raise ValueError("res_file and dcd_file must be specified")

    if not (isinstance(res_file, CharmmFile) and
            isinstance(dcd_file, CharmmFile)):
        raise ValueError("res_file and dcd_file must be CharmmFile objects")

    if dynatype == "heat":
        print("Loading heat dynamics")
        dynamics_dict = {
            'leap': False,
            'verlet': True,
            'cpt': False,
            'new': False,
            'langevin': False,
            'timestep': timestep,
            'start': True,
            'restart': False,
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
            'echeck': -1,
        }

    elif dynatype in ["equil", "prod"]:

        if pmass is None or not restart or str_file is  None:
            current_vals = f"pmass: {pmass}, restart: {restart}, str_file: {str_file}"
            raise ValueError("mass, restart, and str_file must be "
                             "specified for equilibration dynamics"
                             f"current values: {current_vals}")

        tmass = int(pmass * 10)

        if dynatype == "equil":
            print("Loading equilibration dynamics")
            dynamics_dict = {
                'leap': True,
                'verlet': False,
                'cpt': True,
                'new': False,
                'langevin': False,
                'timestep': timestep,
                'start': False,
                'restart': True,
                'nstep': 20 * 1. / timestep,
                'nsavc': 0.01 * 1. / timestep,
                'nsavv': 0,
                'inbfrq': -1,
                'ihbfrq': 50,
                'ilbfrq': 50,
                'imgfrq': 50,
                'ixtfrq': 1000,
                'iunrea': str_file.file_unit,
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
                'ieqfrq': 0,
                'firstt': 300,
                'finalt': 300,
                'tbath': 300,
                'pint pconst pref': 1,
                'pgamma': 5,
                'pmass': pmass,
                'hoover reft': 300,
                'tmass': tmass,
                'iasors': 0,
                'iasvel': 1,
                'ichecw': 0,
                'iscale': 0,  # scale velocities on a restart
                'scale': 1,  # scaling factor for velocity scaling
                'echeck': -1}

        elif dynatype == "prod":
            print("Loading production dynamics")
            dynamics_dict = {
                'leap': True,
                'verlet': False,
                'cpt': True,
                'new': False,
                'langevin': False,
                'timestep': timestep,
                'start': False,
                'restart': True,
                'nstep': 100*1./timestep,
                'nsavc': 0.01*1./timestep,
                'nsavv': 0,
                'inbfrq':-1,
                'ihbfrq': 50,
                'ilbfrq': 50,
                'imgfrq': 50,
                'ixtfrq': 1000,
                'iunrea': str_file.file_unit,
                'iunwri': res_file.file_unit,
                'iuncrd': dcd_file.file_unit,
                'nsavl':  0,  # frequency for saving lambda values in lamda-dynamics
                'iunldm':-1,
                'ilap': -1,
                'ilaf': -1,
                'nprint': 100, # Frequency to write to output
                'iprfrq': 500, # Frequency to calculate averages
                'isvfrq': 1000, # Frequency to save restart file
                'ntrfrq': 1000,
                'ihtfrq': 0,
                'ieqfrq': 0,
                'firstt': 300,
                'finalt': 300,
                'tbath': 300,
                'pint pconst pref': 1,
                'pgamma': 5,
                'pmass': pmass,
                'hoover reft': 300,
                'tmass': tmass,
                'iasors': 0,
                'iasvel': 1,
                'ichecw': 0,
                'iscale': 0,  # scale velocities on a restart
                'scale': 1,  # scaling factor for velocity scaling
                'echeck':-1}
        else:
            raise ValueError("Missing mass and restart file required for equilibration")

    else:
        raise ValueError("Unknown dynamics type: %s" % dynatype)

    if nstep is not None:
        dynamics_dict['nstep'] = nstep

    print("Pychramm dynamics dict:")
    for k,v in dynamics_dict.items():
        print(k, v)

    return dynamics_dict
