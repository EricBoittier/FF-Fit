%nproc=4
%mem=5760MB
%chk=meoh_295.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4136 -0.0013 0.0024
C 0.0248 0.0017 -0.0007
H 1.8691 0.8630 -0.0745
H -0.4017 0.0058 -1.0038
H -0.3519 -0.9151 0.4528
H -0.3903 0.8321 0.5704

