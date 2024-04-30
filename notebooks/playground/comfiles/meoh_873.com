%nproc=4
%mem=5760MB
%chk=meoh_873.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4374 0.0073 0.0185
C -0.0191 -0.0074 0.0190
H 1.7130 0.7983 -0.4902
H -0.1719 0.2956 -1.0169
H -0.3428 -1.0366 0.1743
H -0.3269 0.7004 0.7886

