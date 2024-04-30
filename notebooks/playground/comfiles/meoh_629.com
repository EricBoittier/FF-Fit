%nproc=4
%mem=5760MB
%chk=meoh_629.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4405 0.0016 -0.0424
C -0.0087 -0.0009 0.0065
H 1.6989 0.7946 0.4722
H -0.3307 -1.0417 0.0428
H -0.2666 0.5158 0.9309
H -0.4042 0.5021 -0.8759

