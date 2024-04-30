%nproc=4
%mem=5760MB
%chk=meoh_575.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4113 0.0823 -0.0574
C 0.0299 0.0081 0.0144
H 1.8321 -0.4245 0.6683
H -0.2614 -0.9700 0.3973
H -0.4647 0.7565 0.6336
H -0.4044 0.0201 -0.9852

