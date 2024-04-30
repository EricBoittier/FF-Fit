%nproc=4
%mem=5760MB
%chk=meoh_592.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4359 0.0651 -0.0679
C -0.0030 -0.0194 0.0039
H 1.7562 -0.0052 0.8557
H -0.3176 -1.0035 0.3513
H -0.2849 0.7267 0.7468
H -0.4520 0.2658 -0.9475

