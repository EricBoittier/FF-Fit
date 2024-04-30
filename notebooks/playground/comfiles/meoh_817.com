%nproc=4
%mem=5760MB
%chk=meoh_817.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4156 0.0654 -0.0639
C 0.0336 -0.0108 0.0066
H 1.7961 -0.0927 0.8253
H -0.4745 -0.2693 -0.9225
H -0.3186 -0.7166 0.7588
H -0.4131 0.9533 0.2495

