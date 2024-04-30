%nproc=4
%mem=5760MB
%chk=meoh_110.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4448 -0.0138 0.0061
C -0.0039 0.0017 0.0074
H 1.6211 0.9350 -0.1646
H -0.3712 -1.0216 -0.0710
H -0.3865 0.4960 0.9003
H -0.2909 0.5742 -0.8746

