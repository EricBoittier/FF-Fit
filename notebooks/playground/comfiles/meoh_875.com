%nproc=4
%mem=5760MB
%chk=meoh_875.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4364 0.0102 0.0238
C -0.0102 -0.0079 0.0186
H 1.6918 0.7535 -0.5617
H -0.1877 0.3179 -1.0064
H -0.3728 -1.0278 0.1470
H -0.3514 0.6743 0.7972

