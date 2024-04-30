%nproc=4
%mem=5760MB
%chk=meoh_439.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4347 0.1055 -0.0395
C 0.0010 -0.0035 0.0077
H 1.6893 -0.7052 0.4488
H -0.4003 0.4812 0.8977
H -0.4201 0.4509 -0.8891
H -0.1949 -1.0748 0.0531

