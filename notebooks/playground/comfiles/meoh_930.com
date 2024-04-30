%nproc=4
%mem=5760MB
%chk=meoh_930.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4277 0.1066 -0.0490
C 0.0299 -0.0099 0.0042
H 1.6645 -0.6253 0.5582
H -0.5414 0.8563 -0.3299
H -0.3432 -0.8735 -0.5465
H -0.3383 -0.1453 1.0211

