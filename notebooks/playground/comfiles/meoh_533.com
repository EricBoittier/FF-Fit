%nproc=4
%mem=5760MB
%chk=meoh_533.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4221 0.1051 0.0253
C 0.0360 -0.0014 0.0039
H 1.7146 -0.6702 -0.4981
H -0.3530 -0.7383 0.7066
H -0.5246 0.8983 0.2577
H -0.3801 -0.3564 -0.9389

