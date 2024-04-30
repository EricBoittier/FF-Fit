%nproc=4
%mem=5760MB
%chk=meoh_283.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4520 -0.0078 -0.0426
C -0.0116 0.0013 0.0147
H 1.5545 0.8349 0.4472
H -0.3338 -0.0777 -1.0236
H -0.3463 -0.8296 0.6356
H -0.3244 0.9651 0.4165

