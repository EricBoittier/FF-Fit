%nproc=4
%mem=5760MB
%chk=meoh_480.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4484 -0.0143 -0.0027
C -0.0168 0.0033 0.0006
H 1.6131 0.9513 -0.0333
H -0.3151 -0.1030 1.0437
H -0.3029 0.9761 -0.3992
H -0.3257 -0.8509 -0.6019

