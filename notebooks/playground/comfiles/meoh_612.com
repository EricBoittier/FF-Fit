%nproc=4
%mem=5760MB
%chk=meoh_612.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4358 0.0215 -0.0688
C -0.0046 -0.0030 0.0210
H 1.7040 0.4956 0.7460
H -0.2805 -1.0485 0.1587
H -0.3516 0.6263 0.8406
H -0.3483 0.4058 -0.9291

