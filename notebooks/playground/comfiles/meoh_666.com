%nproc=4
%mem=5760MB
%chk=meoh_666.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4195 -0.0030 0.0107
C 0.0286 0.0012 0.0087
H 1.7414 0.8591 -0.3265
H -0.3568 -0.9920 -0.2218
H -0.4018 0.2646 0.9748
H -0.3952 0.6870 -0.7249

