%nproc=4
%mem=5760MB
%chk=meoh_450.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4165 0.0666 -0.0575
C 0.0168 -0.0110 0.0014
H 1.8425 -0.1327 0.8024
H -0.2577 0.3695 0.9853
H -0.4230 0.6332 -0.7599
H -0.3861 -1.0112 -0.1580

