%nproc=4
%mem=5760MB
%chk=meoh_924.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4145 0.1040 -0.0214
C 0.0247 0.0075 0.0004
H 1.8604 -0.7279 0.2425
H -0.5433 0.7888 -0.5046
H -0.3162 -0.9163 -0.4669
H -0.2879 -0.0996 1.0391

