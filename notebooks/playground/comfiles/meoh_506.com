%nproc=4
%mem=5760MB
%chk=meoh_506.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4333 0.0346 0.0544
C 0.0126 0.0111 -0.0000
H 1.6549 0.2483 -0.8761
H -0.3768 -0.4994 0.8808
H -0.4560 0.9932 -0.0636
H -0.2646 -0.6392 -0.8297

