%nproc=4
%mem=5760MB
%chk=meoh_652.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4435 -0.0050 -0.0044
C -0.0131 -0.0086 0.0033
H 1.6940 0.9418 -0.0410
H -0.3885 -1.0217 -0.1409
H -0.2619 0.3931 0.9856
H -0.3414 0.6527 -0.7985

