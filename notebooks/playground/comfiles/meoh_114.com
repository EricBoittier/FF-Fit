%nproc=4
%mem=5760MB
%chk=meoh_114.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4304 -0.0080 0.0115
C 0.0150 -0.0022 0.0068
H 1.6967 0.9007 -0.2411
H -0.4006 -1.0045 -0.0974
H -0.3996 0.4616 0.9018
H -0.3194 0.5807 -0.8514

