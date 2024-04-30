%nproc=4
%mem=5760MB
%chk=meoh_134.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4402 0.0023 0.0450
C 0.0045 0.0065 -0.0068
H 1.6605 0.6871 -0.6207
H -0.3318 -1.0102 -0.2107
H -0.4062 0.3202 0.9528
H -0.3770 0.6747 -0.7789

