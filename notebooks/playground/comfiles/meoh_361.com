%nproc=4
%mem=5760MB
%chk=meoh_361.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4279 0.0740 -0.0613
C 0.0315 0.0114 0.0042
H 1.6811 -0.3625 0.7788
H -0.5821 0.8200 -0.3930
H -0.3539 -0.8665 -0.5142
H -0.3264 -0.1159 1.0259

