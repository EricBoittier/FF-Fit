%nproc=4
%mem=5760MB
%chk=meoh_529.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4235 0.1002 0.0317
C 0.0184 -0.0092 0.0023
H 1.7762 -0.5631 -0.5977
H -0.3495 -0.7013 0.7598
H -0.4268 0.9592 0.2300
H -0.3556 -0.3903 -0.9480

