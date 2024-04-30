%nproc=4
%mem=5760MB
%chk=meoh_679.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4382 0.0048 0.0281
C -0.0012 -0.0015 0.0094
H 1.6902 0.7384 -0.5710
H -0.3204 -0.9879 -0.3270
H -0.3902 0.2111 1.0052
H -0.3351 0.7639 -0.6911

