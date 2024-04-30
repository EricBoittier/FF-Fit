%nproc=4
%mem=5760MB
%chk=meoh_772.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4161 0.1091 -0.0087
C 0.0273 0.0001 0.0205
H 1.8152 -0.7829 -0.0826
H -0.2725 -0.6369 -0.8117
H -0.3766 -0.4607 0.9219
H -0.5097 0.9317 -0.1584

