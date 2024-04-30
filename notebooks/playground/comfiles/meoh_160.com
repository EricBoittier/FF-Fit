%nproc=4
%mem=5760MB
%chk=meoh_160.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4270 0.0380 0.0640
C 0.0256 0.0084 -0.0141
H 1.7079 0.1737 -0.8651
H -0.3249 -0.9668 -0.3523
H -0.4088 0.1202 0.9792
H -0.4700 0.7553 -0.6343

