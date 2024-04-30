%nproc=4
%mem=5760MB
%chk=meoh_276.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4238 0.0127 -0.0462
C 0.0158 -0.0084 0.0010
H 1.7390 0.6451 0.6329
H -0.4050 -0.1747 -0.9906
H -0.3571 -0.7579 0.6990
H -0.3060 0.9711 0.3547

