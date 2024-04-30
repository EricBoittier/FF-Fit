%nproc=4
%mem=5760MB
%chk=meoh_786.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4362 0.1178 -0.0278
C -0.0097 -0.0201 0.0038
H 1.7486 -0.7562 0.2868
H -0.3559 -0.5714 -0.8705
H -0.2429 -0.5265 0.9405
H -0.3712 1.0082 0.0145

