%nproc=4
%mem=5760MB
%chk=meoh_442.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4136 0.0936 -0.0417
C 0.0264 -0.0012 0.0011
H 1.8149 -0.5643 0.5639
H -0.3687 0.4431 0.9147
H -0.4602 0.4830 -0.8456
H -0.2792 -1.0474 -0.0099

