%nproc=4
%mem=5760MB
%chk=meoh_920.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4241 0.1071 -0.0068
C -0.0087 0.0020 0.0037
H 1.8872 -0.7563 0.0178
H -0.4362 0.8030 -0.5995
H -0.2363 -0.9650 -0.4447
H -0.2566 -0.0204 1.0649

