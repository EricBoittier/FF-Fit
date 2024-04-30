%nproc=4
%mem=5760MB
%chk=meoh_726.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4203 0.0706 0.0483
C 0.0173 -0.0104 -0.0013
H 1.8126 -0.1523 -0.8217
H -0.3503 -0.8641 -0.5707
H -0.3420 -0.1004 1.0239
H -0.4108 0.9059 -0.4078

