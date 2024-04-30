%nproc=4
%mem=5760MB
%chk=meoh_991.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4098 0.0676 0.0389
C 0.0330 -0.0060 0.0229
H 1.8766 -0.1701 -0.7894
H -0.5045 0.7764 0.5586
H -0.3456 0.1696 -0.9841
H -0.3389 -0.9926 0.2992

