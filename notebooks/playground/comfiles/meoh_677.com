%nproc=4
%mem=5760MB
%chk=meoh_677.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4387 0.0029 0.0243
C -0.0092 0.0010 0.0138
H 1.7060 0.7607 -0.5369
H -0.2865 -0.9950 -0.3314
H -0.3762 0.2165 1.0172
H -0.3111 0.7453 -0.7230

