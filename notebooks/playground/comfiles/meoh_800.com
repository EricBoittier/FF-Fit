%nproc=4
%mem=5760MB
%chk=meoh_800.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4244 0.0934 -0.0530
C 0.0116 -0.0018 0.0099
H 1.8055 -0.5292 0.6009
H -0.3442 -0.4841 -0.9006
H -0.2848 -0.6308 0.8493
H -0.4642 0.9690 0.1487

