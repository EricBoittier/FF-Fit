%nproc=4
%mem=5760MB
%chk=meoh_647.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4265 -0.0030 -0.0161
C 0.0125 -0.0055 0.0129
H 1.7544 0.9166 0.0698
H -0.3892 -1.0093 -0.1260
H -0.3445 0.4031 0.9583
H -0.3527 0.5881 -0.8251

