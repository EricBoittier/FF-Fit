%nproc=4
%mem=5760MB
%chk=meoh_892.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4379 0.0492 0.0552
C -0.0218 -0.0015 -0.0074
H 1.7857 0.0968 -0.8598
H -0.3764 0.5539 -0.8757
H -0.2116 -1.0748 -0.0224
H -0.3025 0.4465 0.9457

