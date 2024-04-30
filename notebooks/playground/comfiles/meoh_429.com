%nproc=4
%mem=5760MB
%chk=meoh_429.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4309 0.1112 -0.0084
C 0.0107 0.0025 0.0152
H 1.7297 -0.8178 -0.0987
H -0.4695 0.6024 0.7883
H -0.4031 0.2477 -0.9630
H -0.2384 -1.0423 0.2007

