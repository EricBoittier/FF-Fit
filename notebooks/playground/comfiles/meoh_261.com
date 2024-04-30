%nproc=4
%mem=5760MB
%chk=meoh_261.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4494 0.0441 -0.0717
C -0.0152 0.0050 0.0135
H 1.6442 0.1081 0.8866
H -0.3517 -0.3895 -0.9453
H -0.2374 -0.6938 0.8200
H -0.4216 1.0006 0.1914

