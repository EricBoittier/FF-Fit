%nproc=4
%mem=5760MB
%chk=meoh_343.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4199 0.1071 0.0060
C 0.0007 -0.0068 0.0096
H 1.8383 -0.7608 -0.1737
H -0.3223 0.6910 -0.7630
H -0.2617 -1.0153 -0.3098
H -0.3410 0.2512 1.0120

