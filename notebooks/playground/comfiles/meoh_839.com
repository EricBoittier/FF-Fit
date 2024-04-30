%nproc=4
%mem=5760MB
%chk=meoh_839.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4131 0.0185 -0.0607
C 0.0392 -0.0060 0.0260
H 1.7642 0.5738 0.6665
H -0.3715 -0.0488 -0.9828
H -0.3900 -0.8621 0.5465
H -0.4399 0.8995 0.3984

