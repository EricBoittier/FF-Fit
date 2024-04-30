%nproc=4
%mem=5760MB
%chk=meoh_242.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4247 0.0941 -0.0455
C -0.0030 -0.0160 0.0025
H 1.8315 -0.4802 0.6365
H -0.3501 -0.5542 -0.8795
H -0.2809 -0.5115 0.9328
H -0.3192 1.0268 -0.0234

