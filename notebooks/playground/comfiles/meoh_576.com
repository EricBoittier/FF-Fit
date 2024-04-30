%nproc=4
%mem=5760MB
%chk=meoh_576.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4102 0.0813 -0.0588
C 0.0332 0.0070 0.0148
H 1.8273 -0.4010 0.6856
H -0.2774 -0.9689 0.3878
H -0.4612 0.7514 0.6389
H -0.4100 0.0295 -0.9808

