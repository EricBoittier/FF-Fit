%nproc=4
%mem=5760MB
%chk=meoh_413.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4489 0.0720 0.0619
C -0.0202 -0.0218 -0.0083
H 1.6091 -0.1508 -0.8790
H -0.3100 0.8951 0.5050
H -0.2782 0.0807 -1.0623
H -0.3194 -0.9227 0.5274

