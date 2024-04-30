%nproc=4
%mem=5760MB
%chk=meoh_417.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4301 0.0849 0.0481
C -0.0045 -0.0193 -0.0029
H 1.7666 -0.3638 -0.7556
H -0.3051 0.8484 0.5844
H -0.3144 0.1356 -1.0364
H -0.3350 -0.9525 0.4532

