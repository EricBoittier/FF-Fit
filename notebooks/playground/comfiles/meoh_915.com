%nproc=4
%mem=5760MB
%chk=meoh_915.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4368 0.1158 0.0083
C -0.0211 -0.0178 0.0083
H 1.7746 -0.7662 -0.2532
H -0.3349 0.8059 -0.6329
H -0.2013 -1.0077 -0.4110
H -0.3332 0.1271 1.0425

