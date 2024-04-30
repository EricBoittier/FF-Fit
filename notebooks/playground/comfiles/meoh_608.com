%nproc=4
%mem=5760MB
%chk=meoh_608.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4357 0.0244 -0.0725
C -0.0068 0.0059 0.0246
H 1.7071 0.4079 0.7877
H -0.2335 -1.0513 0.1625
H -0.3718 0.6333 0.8377
H -0.3511 0.3371 -0.9551

