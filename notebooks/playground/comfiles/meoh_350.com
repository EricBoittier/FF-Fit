%nproc=4
%mem=5760MB
%chk=meoh_350.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4296 0.1127 -0.0274
C -0.0151 -0.0211 0.0089
H 1.8033 -0.7478 0.2562
H -0.2643 0.8332 -0.6206
H -0.3005 -0.9987 -0.3797
H -0.2918 0.1614 1.0473

