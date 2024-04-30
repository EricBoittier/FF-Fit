%nproc=4
%mem=5760MB
%chk=meoh_905.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4180 0.1019 0.0343
C 0.0442 -0.0188 0.0159
H 1.6884 -0.5203 -0.6730
H -0.3874 0.6496 -0.7291
H -0.3580 -0.9921 -0.2652
H -0.5181 0.2541 0.9089

