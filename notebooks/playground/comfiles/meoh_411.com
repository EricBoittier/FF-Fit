%nproc=4
%mem=5760MB
%chk=meoh_411.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4497 0.0626 0.0640
C -0.0135 -0.0172 -0.0075
H 1.5711 -0.0309 -0.9040
H -0.3473 0.9038 0.4705
H -0.2830 0.0249 -1.0628
H -0.3311 -0.9019 0.5444

