%nproc=4
%mem=5760MB
%chk=meoh_280.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4457 -0.0008 -0.0466
C -0.0074 -0.0030 0.0109
H 1.5952 0.7685 0.5419
H -0.3484 -0.1105 -1.0188
H -0.3448 -0.8004 0.6731
H -0.3018 0.9761 0.3888

