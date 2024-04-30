%nproc=4
%mem=5760MB
%chk=meoh_715.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4371 0.0542 0.0527
C -0.0015 -0.0086 0.0050
H 1.7093 0.0809 -0.8885
H -0.3100 -0.8994 -0.5423
H -0.3847 -0.0386 1.0249
H -0.3494 0.8843 -0.5143

