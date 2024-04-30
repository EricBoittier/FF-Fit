%nproc=4
%mem=5760MB
%chk=meoh_660.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4329 -0.0065 0.0075
C 0.0160 -0.0048 -0.0029
H 1.6765 0.9175 -0.2103
H -0.4003 -1.0011 -0.1520
H -0.3423 0.3288 0.9710
H -0.4087 0.6993 -0.7184

