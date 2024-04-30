%nproc=4
%mem=5760MB
%chk=meoh_125.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4285 -0.0004 0.0332
C 0.0022 0.0036 -0.0097
H 1.7724 0.7793 -0.4509
H -0.3180 -1.0308 -0.1343
H -0.3244 0.3642 0.9657
H -0.3710 0.6345 -0.8165

