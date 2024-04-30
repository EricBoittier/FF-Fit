%nproc=4
%mem=5760MB
%chk=meoh_294.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4131 -0.0019 -0.0019
C 0.0282 0.0016 0.0011
H 1.8622 0.8689 -0.0265
H -0.4065 -0.0064 -0.9984
H -0.3638 -0.9051 0.4619
H -0.3986 0.8383 0.5542

