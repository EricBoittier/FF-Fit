%nproc=4
%mem=5760MB
%chk=meoh_508.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4290 0.0406 0.0561
C 0.0284 0.0097 0.0001
H 1.6407 0.1692 -0.8922
H -0.4125 -0.5096 0.8510
H -0.4853 0.9702 -0.0398
H -0.3052 -0.6055 -0.8355

