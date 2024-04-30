%nproc=4
%mem=5760MB
%chk=meoh_987.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4332 0.0505 0.0495
C 0.0103 0.0007 0.0170
H 1.7174 0.0709 -0.8882
H -0.5046 0.8066 0.5401
H -0.2651 0.0579 -1.0361
H -0.3612 -0.9609 0.3709

