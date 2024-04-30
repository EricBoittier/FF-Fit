%nproc=4
%mem=5760MB
%chk=meoh_279.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4410 0.0023 -0.0470
C -0.0027 -0.0046 0.0088
H 1.6256 0.7418 0.5690
H -0.3601 -0.1248 -1.0140
H -0.3473 -0.7896 0.6819
H -0.2994 0.9763 0.3802

