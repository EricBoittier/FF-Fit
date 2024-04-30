%nproc=4
%mem=5760MB
%chk=meoh_778.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4196 0.1127 -0.0215
C 0.0166 -0.0072 0.0199
H 1.8027 -0.7841 0.0752
H -0.2481 -0.6222 -0.8402
H -0.3785 -0.4861 0.9158
H -0.4477 0.9747 -0.0717

