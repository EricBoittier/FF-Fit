%nproc=4
%mem=5760MB
%chk=meoh_966.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4166 0.0066 -0.0025
C 0.0197 -0.0204 0.0015
H 1.7918 0.9113 -0.0352
H -0.3003 0.9991 0.2171
H -0.4243 -0.2795 -0.9596
H -0.3274 -0.7069 0.7738

