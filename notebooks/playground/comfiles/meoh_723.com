%nproc=4
%mem=5760MB
%chk=meoh_723.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4274 0.0672 0.0501
C -0.0005 -0.0128 -0.0001
H 1.8112 -0.0863 -0.8385
H -0.3369 -0.8749 -0.5761
H -0.3013 -0.0789 1.0454
H -0.3643 0.9113 -0.4494

