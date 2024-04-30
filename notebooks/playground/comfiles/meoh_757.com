%nproc=4
%mem=5760MB
%chk=meoh_757.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4278 0.1101 0.0237
C 0.0093 -0.0158 -0.0076
H 1.7796 -0.6882 -0.4228
H -0.3576 -0.7601 -0.7143
H -0.3153 -0.2995 0.9935
H -0.4218 0.9725 -0.1677

