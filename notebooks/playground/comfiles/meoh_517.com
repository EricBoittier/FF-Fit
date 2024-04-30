%nproc=4
%mem=5760MB
%chk=meoh_517.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4279 0.0724 0.0493
C 0.0051 -0.0158 0.0063
H 1.7690 -0.1696 -0.8371
H -0.3551 -0.5829 0.8647
H -0.3646 1.0090 0.0392
H -0.3157 -0.4326 -0.9484

