%nproc=4
%mem=5760MB
%chk=meoh_287.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4402 -0.0104 -0.0310
C 0.0058 0.0038 0.0132
H 1.6358 0.8928 0.2953
H -0.3617 -0.0533 -1.0114
H -0.3687 -0.8573 0.5667
H -0.3749 0.9227 0.4592

