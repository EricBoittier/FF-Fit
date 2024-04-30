%nproc=4
%mem=5760MB
%chk=meoh_716.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4379 0.0563 0.0529
C -0.0065 -0.0099 0.0041
H 1.7227 0.0601 -0.8849
H -0.3100 -0.8975 -0.5511
H -0.3603 -0.0435 1.0345
H -0.3399 0.8909 -0.5113

