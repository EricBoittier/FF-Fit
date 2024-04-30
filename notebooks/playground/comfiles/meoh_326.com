%nproc=4
%mem=5760MB
%chk=meoh_326.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4421 0.0683 0.0537
C -0.0241 -0.0008 -0.0028
H 1.7499 -0.1964 -0.8384
H -0.3074 0.4773 -0.9405
H -0.2141 -1.0734 0.0367
H -0.3726 0.5034 0.8985

