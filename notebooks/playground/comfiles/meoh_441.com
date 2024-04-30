%nproc=4
%mem=5760MB
%chk=meoh_441.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4203 0.0978 -0.0412
C 0.0184 -0.0017 0.0033
H 1.7745 -0.6160 0.5294
H -0.3802 0.4553 0.9091
H -0.4481 0.4726 -0.8602
H -0.2502 -1.0580 0.0116

