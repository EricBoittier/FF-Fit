%nproc=4
%mem=5760MB
%chk=meoh_910.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4304 0.1155 0.0223
C 0.0171 -0.0272 0.0134
H 1.6626 -0.6870 -0.4901
H -0.3502 0.7513 -0.6553
H -0.2862 -1.0104 -0.3463
H -0.4752 0.2224 0.9533

