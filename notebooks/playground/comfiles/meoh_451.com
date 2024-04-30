%nproc=4
%mem=5760MB
%chk=meoh_451.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4234 0.0637 -0.0615
C 0.0079 -0.0124 0.0039
H 1.8061 -0.0811 0.8291
H -0.2499 0.3594 0.9956
H -0.4046 0.6586 -0.7495
H -0.3810 -1.0155 -0.1710

