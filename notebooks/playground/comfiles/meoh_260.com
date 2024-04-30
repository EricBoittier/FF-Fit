%nproc=4
%mem=5760MB
%chk=meoh_260.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4508 0.0462 -0.0731
C -0.0151 0.0052 0.0149
H 1.6285 0.0730 0.8904
H -0.3475 -0.3977 -0.9418
H -0.2403 -0.6844 0.8284
H -0.4306 0.9991 0.1807

