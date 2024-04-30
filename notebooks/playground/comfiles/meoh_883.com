%nproc=4
%mem=5760MB
%chk=meoh_883.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4250 0.0233 0.0469
C 0.0398 -0.0079 0.0024
H 1.6560 0.4943 -0.7809
H -0.4140 0.4464 -0.8784
H -0.4191 -0.9939 0.0751
H -0.4570 0.5634 0.7865

