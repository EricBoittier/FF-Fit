%nproc=4
%mem=5760MB
%chk=meoh_377.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4161 0.0161 -0.0507
C 0.0298 -0.0122 0.0054
H 1.7389 0.6139 0.6557
H -0.2929 1.0192 -0.1368
H -0.4325 -0.6304 -0.7642
H -0.3871 -0.3281 0.9616

