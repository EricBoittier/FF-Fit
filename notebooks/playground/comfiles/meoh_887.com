%nproc=4
%mem=5760MB
%chk=meoh_887.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4298 0.0329 0.0542
C 0.0195 -0.0053 -0.0070
H 1.7011 0.3275 -0.8402
H -0.4388 0.5104 -0.8509
H -0.3260 -1.0378 0.0436
H -0.4046 0.5260 0.8451

