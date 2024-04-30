%nproc=4
%mem=5760MB
%chk=meoh_245.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4133 0.0847 -0.0470
C 0.0151 -0.0113 -0.0004
H 1.8760 -0.3834 0.6792
H -0.3784 -0.5307 -0.8742
H -0.3076 -0.5285 0.9032
H -0.3431 1.0180 0.0175

