%nproc=4
%mem=5760MB
%chk=meoh_503.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4376 0.0275 0.0495
C -0.0113 0.0082 0.0022
H 1.7064 0.3650 -0.8304
H -0.3189 -0.4711 0.9316
H -0.3944 1.0225 -0.1101
H -0.2177 -0.6663 -0.8287

