%nproc=4
%mem=5760MB
%chk=meoh_996.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4138 0.0951 0.0336
C 0.0188 -0.0178 0.0073
H 1.8569 -0.4750 -0.6291
H -0.4016 0.7240 0.6864
H -0.3815 0.2744 -0.9635
H -0.2798 -1.0351 0.2605

