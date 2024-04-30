%nproc=4
%mem=5760MB
%chk=meoh_718.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4376 0.0600 0.0527
C -0.0127 -0.0119 0.0026
H 1.7525 0.0185 -0.8745
H -0.3126 -0.8925 -0.5655
H -0.3196 -0.0528 1.0477
H -0.3293 0.9013 -0.5012

