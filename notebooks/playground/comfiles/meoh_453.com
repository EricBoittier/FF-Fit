%nproc=4
%mem=5760MB
%chk=meoh_453.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4379 0.0573 -0.0692
C -0.0088 -0.0138 0.0091
H 1.7223 0.0228 0.8681
H -0.2456 0.3345 1.0145
H -0.3705 0.7059 -0.7253
H -0.3659 -1.0235 -0.1934

