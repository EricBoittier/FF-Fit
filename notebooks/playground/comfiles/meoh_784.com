%nproc=4
%mem=5760MB
%chk=meoh_784.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4355 0.1184 -0.0270
C -0.0107 -0.0188 0.0081
H 1.7507 -0.7716 0.2356
H -0.3108 -0.5917 -0.8693
H -0.2713 -0.5176 0.9416
H -0.3677 1.0110 -0.0018

