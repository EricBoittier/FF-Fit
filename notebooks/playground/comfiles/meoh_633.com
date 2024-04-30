%nproc=4
%mem=5760MB
%chk=meoh_633.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4400 -0.0057 -0.0408
C -0.0082 0.0058 0.0133
H 1.6787 0.8403 0.3925
H -0.2755 -1.0509 0.0075
H -0.3302 0.4891 0.9356
H -0.3742 0.5285 -0.8704

