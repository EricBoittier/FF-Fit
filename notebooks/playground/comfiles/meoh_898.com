%nproc=4
%mem=5760MB
%chk=meoh_898.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4283 0.0730 0.0470
C -0.0067 -0.0019 0.0056
H 1.8096 -0.1953 -0.8150
H -0.3428 0.5642 -0.8631
H -0.2523 -1.0530 -0.1459
H -0.3473 0.3337 0.9852

