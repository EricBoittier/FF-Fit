%nproc=4
%mem=5760MB
%chk=meoh_546.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4287 0.1108 -0.0052
C -0.0128 -0.0076 0.0119
H 1.8173 -0.7850 -0.0901
H -0.2140 -0.8564 0.6655
H -0.4081 0.9412 0.3747
H -0.2622 -0.1820 -1.0348

