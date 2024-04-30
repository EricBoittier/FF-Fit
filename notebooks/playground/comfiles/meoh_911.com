%nproc=4
%mem=5760MB
%chk=meoh_911.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4330 0.1167 0.0196
C 0.0076 -0.0270 0.0123
H 1.6755 -0.7106 -0.4466
H -0.3403 0.7680 -0.6474
H -0.2637 -1.0141 -0.3619
H -0.4495 0.2110 0.9728

