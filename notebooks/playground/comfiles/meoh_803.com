%nproc=4
%mem=5760MB
%chk=meoh_803.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4318 0.0903 -0.0625
C -0.0051 -0.0030 0.0196
H 1.7759 -0.4711 0.6634
H -0.2667 -0.4659 -0.9319
H -0.2895 -0.6671 0.8357
H -0.4261 0.9916 0.1665

