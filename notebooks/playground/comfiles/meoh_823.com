%nproc=4
%mem=5760MB
%chk=meoh_823.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4318 0.0516 -0.0625
C 0.0020 -0.0062 -0.0065
H 1.7663 0.0956 0.8576
H -0.4796 -0.2594 -0.9510
H -0.2377 -0.7481 0.7552
H -0.3400 0.9529 0.3824

