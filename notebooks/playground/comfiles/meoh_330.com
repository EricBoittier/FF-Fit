%nproc=4
%mem=5760MB
%chk=meoh_330.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4430 0.0797 0.0549
C -0.0028 0.0057 -0.0120
H 1.6558 -0.4019 -0.7717
H -0.4255 0.5112 -0.8803
H -0.2083 -1.0648 -0.0057
H -0.4347 0.4075 0.9045

