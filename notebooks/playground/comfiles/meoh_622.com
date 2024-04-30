%nproc=4
%mem=5760MB
%chk=meoh_622.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4234 0.0167 -0.0470
C 0.0238 -0.0152 0.0006
H 1.7697 0.6792 0.5868
H -0.4473 -0.9928 0.1036
H -0.2819 0.5556 0.8774
H -0.4578 0.4592 -0.8544

