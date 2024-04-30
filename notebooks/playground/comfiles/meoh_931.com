%nproc=4
%mem=5760MB
%chk=meoh_931.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4319 0.1064 -0.0532
C 0.0232 -0.0140 0.0056
H 1.6371 -0.5933 0.6017
H -0.5170 0.8793 -0.3079
H -0.3326 -0.8702 -0.5674
H -0.3339 -0.1518 1.0262

