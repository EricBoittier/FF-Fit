%nproc=4
%mem=5760MB
%chk=meoh_10.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4290 -0.0099 -0.0003
C 0.0106 0.0019 0.0001
H 1.7228 0.9251 0.0002
H -0.3665 -1.0208 -0.0000
H -0.3595 0.5139 0.8883
H -0.3595 0.5137 -0.8883

