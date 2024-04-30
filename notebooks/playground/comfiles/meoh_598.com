%nproc=4
%mem=5760MB
%chk=meoh_598.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4184 0.0473 -0.0643
C 0.0272 -0.0012 0.0035
H 1.8110 0.1569 0.8270
H -0.3294 -0.9976 0.2647
H -0.3681 0.6479 0.7849
H -0.4929 0.2409 -0.9233

