%nproc=4
%mem=5760MB
%chk=meoh_900.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4226 0.0815 0.0436
C 0.0126 -0.0055 0.0100
H 1.7845 -0.2931 -0.7866
H -0.3567 0.5780 -0.8334
H -0.2954 -1.0328 -0.1848
H -0.4045 0.3048 0.9680

