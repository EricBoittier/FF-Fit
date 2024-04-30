%nproc=4
%mem=5760MB
%chk=meoh_132.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4431 -0.0004 0.0443
C -0.0046 0.0084 -0.0107
H 1.6623 0.7145 -0.5892
H -0.3063 -1.0243 -0.1863
H -0.3720 0.3281 0.9645
H -0.3762 0.6732 -0.7904

