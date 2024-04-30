%nproc=4
%mem=5760MB
%chk=meoh_707.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4252 0.0383 0.0474
C 0.0291 0.0008 0.0127
H 1.6897 0.2465 -0.8731
H -0.2933 -0.9155 -0.4819
H -0.4952 0.0077 0.9683
H -0.4107 0.8300 -0.5414

