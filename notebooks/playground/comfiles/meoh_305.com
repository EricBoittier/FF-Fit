%nproc=4
%mem=5760MB
%chk=meoh_305.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4494 -0.0031 0.0423
C -0.0140 0.0044 -0.0116
H 1.5897 0.7685 -0.5456
H -0.3200 0.2053 -1.0383
H -0.3063 -0.9796 0.3549
H -0.3441 0.7877 0.6708

