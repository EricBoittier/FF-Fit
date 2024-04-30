%nproc=4
%mem=5760MB
%chk=meoh_738.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4394 0.0877 0.0421
C -0.0050 -0.0048 0.0094
H 1.6850 -0.4183 -0.7605
H -0.2209 -0.8327 -0.6660
H -0.3856 -0.2267 1.0064
H -0.4073 0.9278 -0.3862

