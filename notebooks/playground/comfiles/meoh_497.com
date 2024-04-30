%nproc=4
%mem=5760MB
%chk=meoh_497.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4291 0.0188 0.0352
C -0.0112 -0.0076 0.0099
H 1.8078 0.5800 -0.6733
H -0.3069 -0.3572 0.9991
H -0.3188 1.0155 -0.2066
H -0.2727 -0.6612 -0.8222

