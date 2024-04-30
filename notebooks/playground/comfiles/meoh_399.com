%nproc=4
%mem=5760MB
%chk=meoh_399.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4152 0.0076 0.0257
C 0.0239 0.0156 0.0139
H 1.8286 0.6571 -0.5808
H -0.4760 0.9467 0.2811
H -0.2989 -0.3467 -0.9621
H -0.3435 -0.7787 0.6636

