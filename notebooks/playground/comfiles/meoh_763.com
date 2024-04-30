%nproc=4
%mem=5760MB
%chk=meoh_763.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4411 0.1158 0.0159
C -0.0132 -0.0136 -0.0018
H 1.7341 -0.7640 -0.3013
H -0.3138 -0.7268 -0.7694
H -0.2522 -0.3800 0.9965
H -0.4253 0.9795 -0.1810

