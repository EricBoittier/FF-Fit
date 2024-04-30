%nproc=4
%mem=5760MB
%chk=meoh_409.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4442 0.0521 0.0628
C 0.0011 -0.0097 -0.0042
H 1.5744 0.0935 -0.9077
H -0.3989 0.9030 0.4375
H -0.2989 -0.0451 -1.0515
H -0.3533 -0.8782 0.5509

