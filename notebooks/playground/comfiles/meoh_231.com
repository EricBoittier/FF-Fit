%nproc=4
%mem=5760MB
%chk=meoh_231.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4318 0.1064 -0.0332
C 0.0201 -0.0018 0.0042
H 1.7007 -0.7290 0.4028
H -0.3446 -0.6581 -0.7860
H -0.3368 -0.4314 0.9402
H -0.5261 0.9352 -0.1041

