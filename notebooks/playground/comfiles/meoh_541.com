%nproc=4
%mem=5760MB
%chk=meoh_541.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4351 0.1101 0.0069
C -0.0079 -0.0024 0.0116
H 1.7306 -0.7863 -0.2570
H -0.2053 -0.8339 0.6882
H -0.4865 0.9286 0.3153
H -0.2650 -0.2431 -1.0199

