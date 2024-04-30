%nproc=4
%mem=5760MB
%chk=meoh_419.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4178 0.0885 0.0381
C 0.0123 -0.0134 0.0015
H 1.8449 -0.4521 -0.6590
H -0.3352 0.8101 0.6254
H -0.3487 0.1419 -1.0151
H -0.3522 -0.9596 0.4014

