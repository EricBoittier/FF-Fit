%nproc=4
%mem=5760MB
%chk=meoh_405.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4236 0.0313 0.0513
C 0.0336 0.0069 0.0063
H 1.6797 0.3410 -0.8426
H -0.4934 0.8886 0.3710
H -0.3298 -0.1937 -1.0016
H -0.3945 -0.8294 0.5589

