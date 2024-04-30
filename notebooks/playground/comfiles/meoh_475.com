%nproc=4
%mem=5760MB
%chk=meoh_475.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4299 -0.0073 -0.0176
C -0.0096 0.0050 0.0045
H 1.7800 0.8815 0.2015
H -0.2639 -0.0603 1.0624
H -0.3591 0.9052 -0.5009
H -0.2812 -0.8850 -0.5632

