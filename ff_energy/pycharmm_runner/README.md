
This is a test directory containing the latest version to use PhysNet model potentials in CHARMM via the PyCHARMM API.
Read the mlpot.info file for detailed description of the modules functionality.
The program suite is tested for python3.6 and newer.

Installation:

A. Copy the directory to your system.

B. Compile CHARMM and PyCHARMM:

  1. go to physnet_pycharmm_c47a2 and configure

    cd physnet_pycharmm_c47a2
    ./configure --as-library
    
  2. go to build/cmake and compile CHARMM, note that it does not produce a 
     charmm executable ('--as-library')
  
    cd build/cmake
    make -j4
    make install
    
  3. go to pycharmm (physnet_pycharmm_c47a2/tool/pycharmm) and start setup.py install
  
    cd ../../tool/pycharmm
    python setup.py install
    
  4. Check if physnet_pycharmm_c47a2/lib/libcharmm.so exist (if not PyCHARMM won't work)
  
C. Add directories to you local envrionmental variables (or in e.g. .bashrc, .zshrc ...)

  # PyCHARMM
  export PYCHARMMPATH={/your/path/to/here}/physnet_pycharmm_c47a2
  export PYTHONPATH=$PYCHARMMPATH/tool/pycharmm:$PYTHONPATH
  export CHARMM_LIB_DIR=$PYCHARMMPATH/lib/
  export CHARMM_DATA_DIR=$PYCHARMMPATH/test/data/
  
  # PhysNet in PyCHARMM
  export PYTHONPATH={/your/path/to/here}/PhysNet:$PYTHONPATH


D. If not already done, install python packages required by PhysNet
  
  - tensorflow (>=2.8)
  - numpy (>= 1.19.4)
  - Atomic Simulation Environment (>=3.x), just for the example Scripts

Examples:

The 'examples' directory contain two systems to test the functionality by executing
the respective Script_...py files.
