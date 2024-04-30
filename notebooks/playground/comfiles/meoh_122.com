%nproc=4
%mem=5760MB
%chk=meoh_122.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4190 0.0009 0.0262
C 0.0175 -0.0009 -0.0045
H 1.8067 0.8006 -0.3870
H -0.3615 -1.0156 -0.1264
H -0.3482 0.3871 0.9461
H -0.3707 0.6093 -0.8200

