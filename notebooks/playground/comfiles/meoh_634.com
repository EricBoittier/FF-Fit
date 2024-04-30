%nproc=4
%mem=5760MB
%chk=meoh_634.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4378 -0.0068 -0.0401
C -0.0042 0.0067 0.0148
H 1.6815 0.8489 0.3708
H -0.2723 -1.0498 -0.0019
H -0.3508 0.4810 0.9329
H -0.3715 0.5332 -0.8661

