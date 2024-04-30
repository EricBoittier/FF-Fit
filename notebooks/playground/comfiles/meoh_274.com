%nproc=4
%mem=5760MB
%chk=meoh_274.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4140 0.0195 -0.0451
C 0.0267 -0.0093 -0.0037
H 1.8075 0.5703 0.6636
H -0.4324 -0.2108 -0.9716
H -0.3586 -0.7406 0.7067
H -0.3187 0.9669 0.3367

