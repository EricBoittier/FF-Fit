%nproc=4
%mem=5760MB
%chk=meoh_686.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4289 0.0150 0.0395
C 0.0271 -0.0097 -0.0041
H 1.6949 0.6380 -0.6688
H -0.4304 -0.9470 -0.3207
H -0.4013 0.1753 0.9809
H -0.4082 0.7965 -0.5945

