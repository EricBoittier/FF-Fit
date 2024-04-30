%nproc=4
%mem=5760MB
%chk=meoh_478.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4439 -0.0129 -0.0086
C -0.0207 0.0046 0.0022
H 1.6748 0.9370 0.0609
H -0.2804 -0.0871 1.0569
H -0.3149 0.9549 -0.4431
H -0.2929 -0.8693 -0.5897

