%nproc=4
%mem=5760MB
%chk=meoh_848.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4393 0.0038 -0.0415
C -0.0164 -0.0056 0.0130
H 1.7387 0.8050 0.4370
H -0.3112 0.0180 -1.0361
H -0.3342 -0.9238 0.5070
H -0.2836 0.8923 0.5701

