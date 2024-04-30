%nproc=4
%mem=5760MB
%chk=meoh_171.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4340 0.0631 0.0525
C -0.0127 -0.0069 0.0050
H 1.8062 -0.0861 -0.8418
H -0.2642 -0.9504 -0.4794
H -0.3967 0.0626 1.0228
H -0.2972 0.8396 -0.6199

