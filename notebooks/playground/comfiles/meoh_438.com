%nproc=4
%mem=5760MB
%chk=meoh_438.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4414 0.1088 -0.0381
C -0.0069 -0.0045 0.0097
H 1.6507 -0.7410 0.4029
H -0.4094 0.4944 0.8914
H -0.4066 0.4381 -0.9027
H -0.1720 -1.0801 0.0725

