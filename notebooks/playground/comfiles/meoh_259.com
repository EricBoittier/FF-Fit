%nproc=4
%mem=5760MB
%chk=meoh_259.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4507 0.0483 -0.0738
C -0.0130 0.0052 0.0156
H 1.6210 0.0383 0.8913
H -0.3466 -0.4057 -0.9373
H -0.2470 -0.6736 0.8357
H -0.4392 0.9964 0.1705

