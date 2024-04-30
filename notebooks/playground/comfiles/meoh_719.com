%nproc=4
%mem=5760MB
%chk=meoh_719.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4365 0.0617 0.0524
C -0.0133 -0.0126 0.0020
H 1.7675 -0.0023 -0.8679
H -0.3157 -0.8894 -0.5706
H -0.3055 -0.0575 1.0512
H -0.3294 0.9052 -0.4938

