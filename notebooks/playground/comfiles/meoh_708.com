%nproc=4
%mem=5760MB
%chk=meoh_708.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4260 0.0399 0.0481
C 0.0296 0.0001 0.0122
H 1.6834 0.2261 -0.8790
H -0.2999 -0.9116 -0.4861
H -0.4991 0.0016 0.9654
H -0.4122 0.8337 -0.5338

