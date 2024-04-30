%nproc=4
%mem=5760MB
%chk=meoh_463.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4335 0.0198 -0.0669
C 0.0185 0.0012 0.0152
H 1.6599 0.5141 0.7485
H -0.3742 0.1224 1.0248
H -0.4323 0.7818 -0.5976
H -0.3683 -0.9612 -0.3200

