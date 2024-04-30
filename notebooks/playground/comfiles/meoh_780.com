%nproc=4
%mem=5760MB
%chk=meoh_780.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4258 0.1153 -0.0241
C 0.0049 -0.0118 0.0166
H 1.7827 -0.7845 0.1292
H -0.2549 -0.6168 -0.8521
H -0.3468 -0.4974 0.9268
H -0.4118 0.9936 -0.0446

