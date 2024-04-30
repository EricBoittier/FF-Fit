%nproc=4
%mem=5760MB
%chk=meoh_397.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4229 0.0012 0.0184
C 0.0056 0.0130 0.0127
H 1.8226 0.7482 -0.4743
H -0.4288 0.9844 0.2490
H -0.2830 -0.3689 -0.9666
H -0.3045 -0.7527 0.7237

