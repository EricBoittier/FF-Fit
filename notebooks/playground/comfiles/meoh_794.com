%nproc=4
%mem=5760MB
%chk=meoh_794.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4198 0.1032 -0.0360
C 0.0261 -0.0089 -0.0041
H 1.8116 -0.6382 0.4712
H -0.4692 -0.4917 -0.8466
H -0.2526 -0.5707 0.8874
H -0.4791 0.9529 0.0832

