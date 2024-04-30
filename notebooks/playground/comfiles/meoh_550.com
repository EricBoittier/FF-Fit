%nproc=4
%mem=5760MB
%chk=meoh_550.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4171 0.1119 -0.0136
C 0.0125 -0.0117 0.0088
H 1.8513 -0.7649 0.0433
H -0.3037 -0.8511 0.6281
H -0.3856 0.9137 0.4253
H -0.3444 -0.1492 -1.0119

