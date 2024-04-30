%nproc=4
%mem=5760MB
%chk=meoh_939.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4348 0.0867 -0.0658
C -0.0153 -0.0265 0.0108
H 1.7317 -0.2090 0.8202
H -0.3121 0.9913 -0.2426
H -0.2978 -0.7744 -0.7302
H -0.2545 -0.2825 1.0429

