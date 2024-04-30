%nproc=4
%mem=5760MB
%chk=meoh_809.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4322 0.0825 -0.0721
C -0.0042 -0.0089 0.0295
H 1.7443 -0.3290 0.7608
H -0.2665 -0.3762 -0.9627
H -0.3107 -0.7144 0.8017
H -0.3912 1.0005 0.1687

