%nproc=4
%mem=5760MB
%chk=meoh_481.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4486 -0.0142 0.0003
C -0.0116 0.0022 0.0001
H 1.5908 0.9521 -0.0799
H -0.3371 -0.1109 1.0342
H -0.3028 0.9823 -0.3776
H -0.3478 -0.8382 -0.6073

