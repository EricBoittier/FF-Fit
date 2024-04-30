%nproc=4
%mem=5760MB
%chk=meoh_341.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4171 0.1038 0.0161
C 0.0184 -0.0007 0.0062
H 1.8046 -0.7421 -0.2920
H -0.3913 0.6394 -0.7752
H -0.2744 -1.0142 -0.2679
H -0.3922 0.2635 0.9807

