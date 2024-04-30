%nproc=4
%mem=5760MB
%chk=meoh_764.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4402 0.1156 0.0136
C -0.0119 -0.0120 0.0008
H 1.7372 -0.7715 -0.2788
H -0.3075 -0.7156 -0.7774
H -0.2551 -0.3944 0.9921
H -0.4344 0.9751 -0.1868

