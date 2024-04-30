%nproc=4
%mem=5760MB
%chk=meoh_918.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4306 0.1107 -0.0006
C -0.0204 -0.0054 0.0056
H 1.8558 -0.7675 -0.0927
H -0.3833 0.8088 -0.6217
H -0.2075 -0.9853 -0.4335
H -0.2713 0.0368 1.0655

