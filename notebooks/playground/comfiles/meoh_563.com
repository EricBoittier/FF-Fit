%nproc=4
%mem=5760MB
%chk=meoh_563.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4412 0.1117 -0.0418
C -0.0093 -0.0185 0.0080
H 1.7259 -0.6915 0.4423
H -0.2435 -0.9284 0.5608
H -0.3787 0.8555 0.5445
H -0.4096 -0.0031 -1.0057

