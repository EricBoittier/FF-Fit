%nproc=4
%mem=5760MB
%chk=meoh_868.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4291 -0.0001 0.0084
C -0.0030 -0.0063 0.0125
H 1.7427 0.8769 -0.2963
H -0.2785 0.2657 -1.0064
H -0.3143 -1.0203 0.2637
H -0.3387 0.7407 0.7318

