%nproc=4
%mem=5760MB
%chk=meoh_482.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4476 -0.0136 0.0032
C -0.0047 0.0007 -0.0003
H 1.5762 0.9494 -0.1258
H -0.3605 -0.1190 1.0231
H -0.3060 0.9855 -0.3572
H -0.3721 -0.8228 -0.6126

