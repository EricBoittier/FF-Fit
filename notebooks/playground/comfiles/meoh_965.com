%nproc=4
%mem=5760MB
%chk=meoh_965.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4238 0.0033 -0.0078
C 0.0110 -0.0166 0.0020
H 1.7545 0.9251 0.0290
H -0.3006 1.0061 0.2142
H -0.4075 -0.3094 -0.9610
H -0.3180 -0.6911 0.7925

