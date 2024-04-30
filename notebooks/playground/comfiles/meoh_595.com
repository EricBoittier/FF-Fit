%nproc=4
%mem=5760MB
%chk=meoh_595.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4255 0.0563 -0.0650
C 0.0152 -0.0107 0.0013
H 1.7950 0.0770 0.8425
H -0.3323 -0.9961 0.3115
H -0.3268 0.6846 0.7679
H -0.4855 0.2530 -0.9303

