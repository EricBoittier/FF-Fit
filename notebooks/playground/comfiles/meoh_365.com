%nproc=4
%mem=5760MB
%chk=meoh_365.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4356 0.0542 -0.0644
C -0.0050 0.0176 0.0063
H 1.7312 -0.1271 0.8523
H -0.5294 0.8962 -0.3694
H -0.2513 -0.8460 -0.6116
H -0.2197 -0.2083 1.0508

