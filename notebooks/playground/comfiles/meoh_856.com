%nproc=4
%mem=5760MB
%chk=meoh_856.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4376 -0.0091 -0.0160
C 0.0101 0.0024 -0.0073
H 1.6647 0.9250 0.1748
H -0.4794 0.1033 -0.9760
H -0.3382 -0.9212 0.4550
H -0.3273 0.7942 0.6616

