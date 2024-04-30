%nproc=4
%mem=5760MB
%chk=meoh_466.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4159 0.0124 -0.0531
C 0.0364 0.0043 0.0107
H 1.7693 0.6247 0.6257
H -0.3771 0.0622 1.0176
H -0.4683 0.7764 -0.5701
H -0.3726 -0.9258 -0.3840

