%nproc=4
%mem=5760MB
%chk=meoh_422.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4056 0.0929 0.0227
C 0.0335 -0.0032 0.0075
H 1.9068 -0.5712 -0.4952
H -0.3973 0.7391 0.6795
H -0.3973 0.1482 -0.9822
H -0.3635 -0.9681 0.3228

