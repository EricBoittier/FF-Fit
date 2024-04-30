%nproc=4
%mem=5760MB
%chk=meoh_324.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4355 0.0623 0.0506
C -0.0210 -0.0042 0.0040
H 1.8018 -0.0862 -0.8462
H -0.2680 0.4509 -0.9552
H -0.2500 -1.0690 0.0470
H -0.3601 0.5508 0.8787

