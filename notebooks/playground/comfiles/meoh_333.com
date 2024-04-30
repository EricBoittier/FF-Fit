%nproc=4
%mem=5760MB
%chk=meoh_333.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4338 0.0875 0.0496
C 0.0238 0.0092 -0.0119
H 1.6407 -0.5336 -0.6797
H -0.4984 0.5199 -0.8210
H -0.2462 -1.0455 -0.0642
H -0.4803 0.3453 0.8941

