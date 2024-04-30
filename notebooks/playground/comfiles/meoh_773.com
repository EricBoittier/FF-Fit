%nproc=4
%mem=5760MB
%chk=meoh_773.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4145 0.1090 -0.0113
C 0.0295 0.0001 0.0216
H 1.8202 -0.7820 -0.0569
H -0.2673 -0.6325 -0.8151
H -0.3882 -0.4640 0.9150
H -0.5082 0.9334 -0.1452

