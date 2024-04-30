%nproc=4
%mem=5760MB
%chk=meoh_559.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4281 0.1165 -0.0330
C 0.0184 -0.0217 0.0052
H 1.7515 -0.7339 0.3316
H -0.3432 -0.8798 0.5719
H -0.3824 0.8580 0.5086
H -0.4551 -0.0487 -0.9761

