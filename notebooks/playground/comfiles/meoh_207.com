%nproc=4
%mem=5760MB
%chk=meoh_207.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4238 0.1150 0.0163
C 0.0303 -0.0196 0.0039
H 1.7303 -0.7640 -0.2903
H -0.3690 -0.7681 -0.6805
H -0.4446 -0.1921 0.9697
H -0.4184 0.9164 -0.3290

