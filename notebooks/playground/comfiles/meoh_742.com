%nproc=4
%mem=5760MB
%chk=meoh_742.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4380 0.0918 0.0366
C -0.0107 -0.0059 0.0157
H 1.7224 -0.4825 -0.7049
H -0.2096 -0.7953 -0.7092
H -0.3658 -0.2564 1.0153
H -0.3860 0.9318 -0.3941

