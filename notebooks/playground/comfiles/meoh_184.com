%nproc=4
%mem=5760MB
%chk=meoh_184.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4204 0.0843 0.0526
C 0.0343 -0.0088 -0.0052
H 1.7063 -0.3940 -0.7537
H -0.3485 -0.8818 -0.5339
H -0.4293 -0.0247 0.9812
H -0.4241 0.8522 -0.4916

