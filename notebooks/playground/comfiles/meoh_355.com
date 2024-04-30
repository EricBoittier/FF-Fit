%nproc=4
%mem=5760MB
%chk=meoh_355.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4254 0.1019 -0.0470
C 0.0268 -0.0120 0.0054
H 1.7074 -0.6318 0.5384
H -0.4429 0.8353 -0.4941
H -0.3863 -0.9316 -0.4091
H -0.3644 0.0398 1.0215

