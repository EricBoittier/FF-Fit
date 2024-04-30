%nproc=4
%mem=5760MB
%chk=meoh_398.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4186 0.0043 0.0220
C 0.0151 0.0146 0.0134
H 1.8297 0.7035 -0.5282
H -0.4539 0.9657 0.2655
H -0.2899 -0.3597 -0.9638
H -0.3239 -0.7673 0.6930

