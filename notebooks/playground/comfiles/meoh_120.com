%nproc=4
%mem=5760MB
%chk=meoh_120.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4162 0.0005 0.0218
C 0.0248 -0.0032 -0.0008
H 1.8057 0.8203 -0.3478
H -0.3874 -1.0050 -0.1215
H -0.3689 0.4041 0.9304
H -0.3660 0.5957 -0.8234

