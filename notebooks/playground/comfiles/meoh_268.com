%nproc=4
%mem=5760MB
%chk=meoh_268.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4160 0.0331 -0.0518
C 0.0209 -0.0033 -0.0043
H 1.8394 0.3527 0.7722
H -0.4345 -0.3116 -0.9454
H -0.3053 -0.7210 0.7484
H -0.3642 0.9778 0.2736

