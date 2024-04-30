%nproc=4
%mem=5760MB
%chk=meoh_345.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4239 0.1101 -0.0039
C -0.0144 -0.0128 0.0111
H 1.8526 -0.7700 -0.0521
H -0.2669 0.7407 -0.7351
H -0.2562 -1.0155 -0.3412
H -0.3011 0.2351 1.0331

