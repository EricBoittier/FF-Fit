%nproc=4
%mem=5760MB
%chk=meoh_536.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4267 0.1077 0.0191
C 0.0272 0.0010 0.0068
H 1.6924 -0.7307 -0.4134
H -0.3066 -0.7767 0.6937
H -0.5452 0.8892 0.2743
H -0.3514 -0.3181 -0.9642

