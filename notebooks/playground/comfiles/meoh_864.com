%nproc=4
%mem=5760MB
%chk=meoh_864.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4211 -0.0059 0.0023
C 0.0281 -0.0032 0.0016
H 1.7259 0.9152 -0.1364
H -0.4327 0.2306 -0.9582
H -0.3384 -0.9733 0.3374
H -0.3867 0.7444 0.6776

