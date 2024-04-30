%nproc=4
%mem=5760MB
%chk=meoh_626.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4333 0.0087 -0.0434
C 0.0050 -0.0079 0.0021
H 1.7347 0.7489 0.5239
H -0.3944 -1.0200 0.0683
H -0.2552 0.5326 0.9122
H -0.4366 0.4795 -0.8670

