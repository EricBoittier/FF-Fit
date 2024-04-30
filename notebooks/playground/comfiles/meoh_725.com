%nproc=4
%mem=5760MB
%chk=meoh_725.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4223 0.0694 0.0488
C 0.0115 -0.0114 -0.0010
H 1.8156 -0.1299 -0.8264
H -0.3472 -0.8674 -0.5725
H -0.3261 -0.0927 1.0322
H -0.3951 0.9083 -0.4216

