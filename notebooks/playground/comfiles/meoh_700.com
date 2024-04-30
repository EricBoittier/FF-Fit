%nproc=4
%mem=5760MB
%chk=meoh_700.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4303 0.0305 0.0444
C -0.0034 -0.0010 0.0096
H 1.7525 0.3836 -0.8111
H -0.2589 -0.9507 -0.4605
H -0.3547 0.0524 1.0400
H -0.3419 0.8273 -0.6128

