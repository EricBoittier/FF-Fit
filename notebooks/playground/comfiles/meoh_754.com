%nproc=4
%mem=5760MB
%chk=meoh_754.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4178 0.1043 0.0247
C 0.0253 -0.0130 -0.0023
H 1.8167 -0.6377 -0.4762
H -0.3635 -0.7564 -0.6983
H -0.3622 -0.2762 0.9819
H -0.4374 0.9544 -0.1978

