%nproc=4
%mem=5760MB
%chk=meoh_106.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4454 -0.0147 0.0018
C -0.0071 0.0032 0.0048
H 1.6317 0.9438 -0.0827
H -0.3562 -1.0286 -0.0372
H -0.3686 0.5133 0.8976
H -0.3050 0.5523 -0.8884

