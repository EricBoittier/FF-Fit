%nproc=4
%mem=5760MB
%chk=meoh_649.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4341 -0.0035 -0.0113
C -0.0001 -0.0073 0.0092
H 1.7332 0.9291 0.0253
H -0.3904 -1.0147 -0.1358
H -0.3024 0.4008 0.9736
H -0.3431 0.6127 -0.8190

