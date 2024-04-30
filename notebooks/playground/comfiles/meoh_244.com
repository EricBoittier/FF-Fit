%nproc=4
%mem=5760MB
%chk=meoh_244.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4166 0.0878 -0.0462
C 0.0088 -0.0130 0.0003
H 1.8666 -0.4152 0.6643
H -0.3688 -0.5389 -0.8766
H -0.2977 -0.5231 0.9135
H -0.3310 1.0227 0.0035

