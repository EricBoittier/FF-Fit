%nproc=4
%mem=5760MB
%chk=meoh_137.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4322 0.0083 0.0443
C 0.0190 0.0013 0.0007
H 1.6891 0.6392 -0.6603
H -0.3727 -0.9849 -0.2484
H -0.4501 0.3071 0.9358
H -0.3665 0.6757 -0.7640

