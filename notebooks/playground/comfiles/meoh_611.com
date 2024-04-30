%nproc=4
%mem=5760MB
%chk=meoh_611.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4369 0.0219 -0.0703
C -0.0073 -0.0005 0.0228
H 1.6997 0.4746 0.7582
H -0.2620 -1.0516 0.1591
H -0.3563 0.6300 0.8405
H -0.3428 0.3903 -0.9379

