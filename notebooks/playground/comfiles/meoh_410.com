%nproc=4
%mem=5760MB
%chk=meoh_410.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4476 0.0575 0.0638
C -0.0070 -0.0137 -0.0062
H 1.5675 0.0310 -0.9085
H -0.3720 0.9045 0.4540
H -0.2901 -0.0088 -1.0587
H -0.3413 -0.8904 0.5487

