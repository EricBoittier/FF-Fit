%nproc=4
%mem=5760MB
%chk=meoh_430.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4377 0.1135 -0.0127
C 0.0021 0.0009 0.0156
H 1.6871 -0.8339 -0.0392
H -0.4672 0.5899 0.8037
H -0.3948 0.2716 -0.9628
H -0.2113 -1.0540 0.1884

