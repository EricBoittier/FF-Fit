%nproc=4
%mem=5760MB
%chk=meoh_858.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4305 -0.0100 -0.0102
C 0.0258 0.0021 -0.0086
H 1.6719 0.9335 0.0994
H -0.5113 0.1369 -0.9475
H -0.3503 -0.9225 0.4294
H -0.3641 0.7711 0.6582

