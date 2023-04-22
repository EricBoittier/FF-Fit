%nproc=4
%mem=2000MB
#P CCSD/aug-cc-pVTZ scf(maxcycle=200) opt

Water opt

0 1
O    0.0000    0.0000    0.0000
H    0.0000    0.0000    1.3112
H    1.0354    0.0000   -0.6225

