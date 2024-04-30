%nproc=4
%mem=5760MB
%chk=meoh_922.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4180 0.1046 -0.0136
C 0.0082 0.0067 0.0018
H 1.8895 -0.7425 0.1297
H -0.4940 0.7945 -0.5598
H -0.2763 -0.9412 -0.4549
H -0.2644 -0.0667 1.0546

