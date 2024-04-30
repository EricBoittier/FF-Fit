%nproc=4
%mem=5760MB
%chk=meoh_702.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4279 0.0321 0.0448
C 0.0071 0.0004 0.0116
H 1.7361 0.3453 -0.8312
H -0.2594 -0.9411 -0.4686
H -0.4035 0.0387 1.0205
H -0.3637 0.8268 -0.5949

