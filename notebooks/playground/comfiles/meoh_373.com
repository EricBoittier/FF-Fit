%nproc=4
%mem=5760MB
%chk=meoh_373.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4262 0.0269 -0.0578
C -0.0093 0.0005 0.0063
H 1.8002 0.3809 0.7761
H -0.3266 1.0149 -0.2354
H -0.2872 -0.7365 -0.7471
H -0.2556 -0.3065 1.0228

