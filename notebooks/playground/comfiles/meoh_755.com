%nproc=4
%mem=5760MB
%chk=meoh_755.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4207 0.1063 0.0245
C 0.0208 -0.0140 -0.0046
H 1.8056 -0.6546 -0.4586
H -0.3642 -0.7582 -0.7018
H -0.3481 -0.2824 0.9854
H -0.4328 0.9606 -0.1846

