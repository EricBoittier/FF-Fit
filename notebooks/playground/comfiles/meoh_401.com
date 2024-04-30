%nproc=4
%mem=5760MB
%chk=meoh_401.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4124 0.0145 0.0339
C 0.0369 0.0156 0.0134
H 1.8021 0.5598 -0.6812
H -0.5064 0.9128 0.3103
H -0.3178 -0.3094 -0.9646
H -0.3769 -0.7954 0.6127

