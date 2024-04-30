%nproc=4
%mem=5760MB
%chk=meoh_779.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4226 0.1140 -0.0229
C 0.0108 -0.0095 0.0184
H 1.7930 -0.7847 0.1022
H -0.2499 -0.6199 -0.8462
H -0.3640 -0.4917 0.9212
H -0.4295 0.9847 -0.0577

