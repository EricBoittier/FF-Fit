%nproc=4
%mem=5760MB
%chk=meoh_969.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4030 0.0147 0.0125
C 0.0384 -0.0274 0.0005
H 1.8493 0.8535 -0.2276
H -0.3108 0.9768 0.2408
H -0.4502 -0.2025 -0.9580
H -0.3558 -0.7490 0.7160

