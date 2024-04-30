%nproc=4
%mem=5760MB
%chk=meoh_393.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4426 -0.0089 0.0047
C -0.0243 0.0029 0.0087
H 1.7287 0.8954 -0.2420
H -0.3262 1.0373 0.1730
H -0.2946 -0.3789 -0.9758
H -0.2573 -0.6612 0.8410

