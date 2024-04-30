%nproc=4
%mem=5760MB
%chk=meoh_927.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4170 0.1053 -0.0350
C 0.0379 0.0016 0.0008
H 1.7697 -0.6939 0.4091
H -0.5752 0.8032 -0.4113
H -0.3515 -0.8870 -0.4960
H -0.3267 -0.1281 1.0198

