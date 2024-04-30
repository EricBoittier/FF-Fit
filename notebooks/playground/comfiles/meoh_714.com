%nproc=4
%mem=5760MB
%chk=meoh_714.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4359 0.0521 0.0523
C 0.0043 -0.0073 0.0060
H 1.6978 0.1018 -0.8908
H -0.3106 -0.9009 -0.5330
H -0.4098 -0.0335 1.0139
H -0.3609 0.8770 -0.5164

