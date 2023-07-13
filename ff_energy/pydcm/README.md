# Ideas for Kernel Code
- add random noise to input?
- tests permutations of input?


# Kai's README: PyDCM
Combine all models and improve user friendlines of the (Minimal) Distributed Charge Method (MDCM)

The aid is to combine, extend and simplify the application of the MDCM modules (see e.g. https://github.com/MMunibas/MDCM) in one package avaiable by Python-

- Enable arbitrary mutlidimensional QM scan for molecules with few slected QM programs.
- Fit MDCM model to one or multiple configurations of whole molecules or fragments to match ESP.
- Fit FMDCM model to a reference MDCM model to account the ESP deviation at conformational changes.
- Use the better performance of Fortran code via f2py for time critical steps.

## Requirements
- Python 3.x
- Numpy
- Scipy
- ASE

## Installation

To compile the fortran code of the mdcm module, go to pydcm via terminal and run

python -m numpy.f2py -c -m mdcm mdcm.F90 --debug-capi

The suffix --debug-capi is for debugging purpose and can be omitted.

## Current State

The Script execute.py contains a commented workflow of functions in the current version.
As long as the data directory still contains all output files produces by the QM scan, there
will be no Gaussian jobs submitted.
