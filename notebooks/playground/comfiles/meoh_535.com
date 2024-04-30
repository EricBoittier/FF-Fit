%nproc=4
%mem=5760MB
%chk=meoh_535.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4248 0.1069 0.0213
C 0.0321 0.0007 0.0058
H 1.6962 -0.7127 -0.4425
H -0.3261 -0.7632 0.6959
H -0.5441 0.8879 0.2684
H -0.3652 -0.3318 -0.9532

