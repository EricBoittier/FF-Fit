%nproc=4
%mem=5760MB
%chk=meoh_14.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4306 -0.0103 -0.0003
C 0.0093 0.0018 0.0001
H 1.7220 0.9254 0.0003
H -0.3669 -1.0213 -0.0000
H -0.3591 0.5143 0.8888
H -0.3591 0.5140 -0.8888

