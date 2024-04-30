%nproc=4
%mem=5760MB
%chk=meoh_243.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4205 0.0910 -0.0458
C 0.0026 -0.0146 0.0013
H 1.8515 -0.4477 0.6504
H -0.3592 -0.5467 -0.8785
H -0.2886 -0.5175 0.9235
H -0.3227 1.0257 -0.0102

