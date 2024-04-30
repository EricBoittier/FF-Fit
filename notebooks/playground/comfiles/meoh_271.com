%nproc=4
%mem=5760MB
%chk=meoh_271.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4083 0.0278 -0.0459
C 0.0318 -0.0077 -0.0072
H 1.8610 0.4575 0.7098
H -0.4508 -0.2638 -0.9504
H -0.3429 -0.7255 0.7225
H -0.3418 0.9670 0.3068

