%nproc=4
%mem=5760MB
%chk=meoh_473.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4194 -0.0023 -0.0237
C 0.0058 0.0049 0.0054
H 1.8328 0.8278 0.2935
H -0.2785 -0.0390 1.0568
H -0.3984 0.8655 -0.5275
H -0.2966 -0.8915 -0.5358

