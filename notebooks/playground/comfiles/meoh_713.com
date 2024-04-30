%nproc=4
%mem=5760MB
%chk=meoh_713.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4342 0.0500 0.0518
C 0.0104 -0.0059 0.0070
H 1.6886 0.1226 -0.8919
H -0.3113 -0.9021 -0.5235
H -0.4342 -0.0281 1.0020
H -0.3731 0.8691 -0.5178

