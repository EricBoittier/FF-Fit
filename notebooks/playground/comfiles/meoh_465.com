%nproc=4
%mem=5760MB
%chk=meoh_465.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4211 0.0145 -0.0580
C 0.0323 0.0036 0.0122
H 1.7336 0.5910 0.6704
H -0.3809 0.0810 1.0179
H -0.4603 0.7763 -0.5780
H -0.3739 -0.9364 -0.3614

