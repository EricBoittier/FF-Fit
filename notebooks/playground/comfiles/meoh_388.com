%nproc=4
%mem=5760MB
%chk=meoh_388.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4482 -0.0112 -0.0146
C -0.0103 -0.0107 0.0058
H 1.5896 0.9543 0.0762
H -0.2637 1.0472 0.0743
H -0.4087 -0.3941 -0.9335
H -0.3220 -0.5164 0.9197

