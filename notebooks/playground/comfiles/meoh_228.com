%nproc=4
%mem=5760MB
%chk=meoh_228.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4235 0.1036 -0.0258
C 0.0276 0.0067 0.0008
H 1.7573 -0.7524 0.3155
H -0.3324 -0.6932 -0.7534
H -0.3370 -0.4145 0.9376
H -0.5525 0.9210 -0.1245

