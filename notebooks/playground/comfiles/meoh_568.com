%nproc=4
%mem=5760MB
%chk=meoh_568.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4367 0.0984 -0.0499
C -0.0136 -0.0046 0.0118
H 1.7732 -0.5973 0.5528
H -0.1745 -0.9653 0.5010
H -0.4130 0.8285 0.5904
H -0.3694 0.0109 -1.0183

