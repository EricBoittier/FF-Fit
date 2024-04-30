%nproc=4
%mem=5760MB
%chk=meoh_142.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4223 0.0202 0.0429
C 0.0229 -0.0081 0.0097
H 1.7641 0.5454 -0.7106
H -0.3904 -0.9676 -0.3013
H -0.4562 0.2859 0.9435
H -0.3070 0.6977 -0.7526

