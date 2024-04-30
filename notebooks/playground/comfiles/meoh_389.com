%nproc=4
%mem=5760MB
%chk=meoh_389.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4497 -0.0119 -0.0105
C -0.0180 -0.0083 0.0062
H 1.6094 0.9548 0.0121
H -0.2681 1.0491 0.0932
H -0.3800 -0.3870 -0.9497
H -0.2985 -0.5446 0.9127

