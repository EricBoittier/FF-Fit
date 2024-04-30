%nproc=4
%mem=5760MB
%chk=meoh_877.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4333 0.0131 0.0296
C 0.0053 -0.0086 0.0165
H 1.6715 0.7001 -0.6275
H -0.2343 0.3473 -0.9855
H -0.4046 -1.0128 0.1248
H -0.3871 0.6448 0.7957

