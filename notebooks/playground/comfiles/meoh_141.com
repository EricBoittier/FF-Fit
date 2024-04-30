%nproc=4
%mem=5760MB
%chk=meoh_141.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4231 0.0179 0.0429
C 0.0249 -0.0065 0.0088
H 1.7512 0.5649 -0.7012
H -0.3939 -0.9668 -0.2923
H -0.4628 0.2899 0.9374
H -0.3218 0.6906 -0.7540

