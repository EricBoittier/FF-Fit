%nproc=4
%mem=5760MB
%chk=meoh_166.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4271 0.0500 0.0574
C 0.0125 0.0038 -0.0022
H 1.7849 0.0341 -0.8549
H -0.2996 -0.9523 -0.4223
H -0.4142 0.0795 0.9980
H -0.4142 0.7852 -0.6310

