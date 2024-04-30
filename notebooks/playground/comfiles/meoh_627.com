%nproc=4
%mem=5760MB
%chk=meoh_627.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4361 0.0063 -0.0429
C -0.0003 -0.0056 0.0034
H 1.7223 0.7651 0.5074
H -0.3733 -1.0282 0.0597
H -0.2553 0.5271 0.9196
H -0.4261 0.4868 -0.8709

