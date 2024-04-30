%nproc=4
%mem=5760MB
%chk=meoh_505.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4352 0.0320 0.0531
C 0.0040 0.0108 0.0004
H 1.6688 0.2876 -0.8638
H -0.3565 -0.4922 0.8977
H -0.4365 1.0047 -0.0778
H -0.2454 -0.6516 -0.8285

