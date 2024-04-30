%nproc=4
%mem=5760MB
%chk=meoh_251.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4162 0.0679 -0.0598
C 0.0299 -0.0013 0.0043
H 1.7959 -0.2118 0.7993
H -0.4000 -0.4743 -0.8787
H -0.3356 -0.5706 0.8589
H -0.4377 0.9792 0.0935

