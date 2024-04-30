%nproc=4
%mem=5760MB
%chk=meoh_499.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4341 0.0214 0.0401
C -0.0201 -0.0020 0.0079
H 1.7863 0.5111 -0.7323
H -0.2899 -0.4011 0.9857
H -0.3302 1.0262 -0.1783
H -0.2306 -0.6661 -0.8304

