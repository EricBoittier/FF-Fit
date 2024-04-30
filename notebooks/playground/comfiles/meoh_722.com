%nproc=4
%mem=5760MB
%chk=meoh_722.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4301 0.0659 0.0507
C -0.0056 -0.0131 0.0004
H 1.8040 -0.0650 -0.8457
H -0.3310 -0.8787 -0.5768
H -0.2947 -0.0729 1.0496
H -0.3512 0.9113 -0.4624

