%nproc=4
%mem=5760MB
%chk=meoh_917.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4334 0.1126 0.0024
C -0.0231 -0.0097 0.0065
H 1.8317 -0.7703 -0.1471
H -0.3622 0.8097 -0.6274
H -0.1999 -0.9940 -0.4270
H -0.2872 0.0675 1.0612

