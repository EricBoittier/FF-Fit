%nproc=4
%mem=5760MB
%chk=meoh_694.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4319 0.0258 0.0443
C -0.0049 -0.0074 0.0002
H 1.7600 0.4958 -0.7507
H -0.3344 -0.9634 -0.4066
H -0.2958 0.1049 1.0446
H -0.3411 0.8251 -0.6180

