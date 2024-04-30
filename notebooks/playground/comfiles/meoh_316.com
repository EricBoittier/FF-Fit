%nproc=4
%mem=5760MB
%chk=meoh_316.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4129 0.0352 0.0456
C 0.0358 -0.0131 0.0143
H 1.7749 0.3412 -0.8122
H -0.2958 0.3496 -0.9586
H -0.4531 -0.9813 0.1221
H -0.4193 0.6732 0.7285

