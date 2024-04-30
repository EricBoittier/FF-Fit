%nproc=4
%mem=5760MB
%chk=meoh_665.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4199 -0.0037 0.0102
C 0.0306 0.0003 0.0063
H 1.7320 0.8694 -0.3074
H -0.3704 -0.9908 -0.2063
H -0.4002 0.2736 0.9695
H -0.4060 0.6878 -0.7181

