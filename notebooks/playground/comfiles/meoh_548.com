%nproc=4
%mem=5760MB
%chk=meoh_548.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4227 0.1111 -0.0095
C -0.0019 -0.0095 0.0106
H 1.8415 -0.7749 -0.0231
H -0.2544 -0.8561 0.6490
H -0.3918 0.9306 0.4008
H -0.2970 -0.1648 -1.0272

