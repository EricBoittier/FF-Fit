%nproc=4
%mem=5760MB
%chk=meoh_348.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4287 0.1129 -0.0183
C -0.0222 -0.0195 0.0104
H 1.8352 -0.7659 0.1334
H -0.2383 0.8039 -0.6704
H -0.2726 -1.0100 -0.3693
H -0.2793 0.1976 1.0471

