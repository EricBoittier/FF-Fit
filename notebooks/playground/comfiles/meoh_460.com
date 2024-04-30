%nproc=4
%mem=5760MB
%chk=meoh_460.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4505 0.0301 -0.0760
C -0.0069 -0.0044 0.0179
H 1.5824 0.3783 0.8306
H -0.3349 0.1921 1.0387
H -0.3789 0.7872 -0.6326
H -0.3510 -0.9980 -0.2695

