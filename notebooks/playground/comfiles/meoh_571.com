%nproc=4
%mem=5760MB
%chk=meoh_571.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4245 0.0899 -0.0531
C 0.0043 0.0044 0.0132
H 1.8139 -0.5241 0.6041
H -0.1948 -0.9732 0.4523
H -0.4452 0.7963 0.6123
H -0.3766 0.0070 -1.0081

