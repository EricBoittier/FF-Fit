%nproc=4
%mem=5760MB
%chk=meoh_146.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4252 0.0271 0.0459
C 0.0057 -0.0108 0.0075
H 1.7837 0.4683 -0.7524
H -0.3504 -0.9857 -0.3255
H -0.4056 0.2668 0.9780
H -0.2582 0.7346 -0.7427

