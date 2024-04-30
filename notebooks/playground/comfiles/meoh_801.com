%nproc=4
%mem=5760MB
%chk=meoh_801.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4269 0.0923 -0.0563
C 0.0057 -0.0019 0.0132
H 1.7964 -0.5105 0.6224
H -0.3156 -0.4804 -0.9120
H -0.2867 -0.6430 0.8448
H -0.4518 0.9770 0.1565

