%nproc=4
%mem=5760MB
%chk=meoh_434.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4538 0.1163 -0.0282
C -0.0202 -0.0051 0.0151
H 1.5833 -0.8296 0.1934
H -0.4426 0.5442 0.8565
H -0.3765 0.3670 -0.9455
H -0.1404 -1.0815 0.1378

