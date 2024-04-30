%nproc=4
%mem=5760MB
%chk=meoh_222.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4213 0.1028 -0.0110
C 0.0047 0.0065 -0.0019
H 1.8461 -0.7687 0.1324
H -0.2662 -0.7483 -0.7402
H -0.2711 -0.3768 0.9805
H -0.4659 0.9695 -0.1997

