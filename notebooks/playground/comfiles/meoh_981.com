%nproc=4
%mem=5760MB
%chk=meoh_981.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4594 0.0294 0.0563
C -0.0194 0.0003 -0.0035
H 1.5145 0.4208 -0.8405
H -0.4149 0.8681 0.5243
H -0.2136 -0.0767 -1.0732
H -0.3609 -0.8973 0.5121

