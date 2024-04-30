%nproc=4
%mem=5760MB
%chk=meoh_252.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4207 0.0654 -0.0628
C 0.0265 0.0000 0.0063
H 1.7650 -0.1837 0.8204
H -0.3948 -0.4650 -0.8850
H -0.3297 -0.5816 0.8565
H -0.4480 0.9765 0.1041

