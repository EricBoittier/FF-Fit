%nproc=4
%mem=5760MB
%chk=meoh_711.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4305 0.0457 0.0505
C 0.0215 -0.0031 0.0093
H 1.6783 0.1643 -0.8903
H -0.3107 -0.9045 -0.5057
H -0.4748 -0.0166 0.9797
H -0.3960 0.8528 -0.5210

