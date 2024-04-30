%nproc=4
%mem=5760MB
%chk=meoh_975.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4259 0.0192 0.0410
C 0.0178 -0.0178 -0.0041
H 1.6882 0.7072 -0.6058
H -0.3479 0.9341 0.3811
H -0.3553 -0.1267 -1.0224
H -0.3720 -0.8222 0.6196

