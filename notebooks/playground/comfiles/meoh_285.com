%nproc=4
%mem=5760MB
%chk=meoh_285.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4486 -0.0102 -0.0376
C -0.0054 0.0032 0.0149
H 1.5776 0.8694 0.3750
H -0.3424 -0.0636 -1.0196
H -0.3557 -0.8451 0.6028
H -0.3489 0.9475 0.4371

