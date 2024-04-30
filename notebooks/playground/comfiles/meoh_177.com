%nproc=4
%mem=5760MB
%chk=meoh_177.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4310 0.0770 0.0524
C 0.0016 -0.0157 0.0026
H 1.7471 -0.2351 -0.8212
H -0.3050 -0.9263 -0.5121
H -0.4252 0.0390 1.0041
H -0.2903 0.8721 -0.5585

