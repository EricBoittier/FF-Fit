%nproc=4
%mem=5760MB
%chk=meoh_263.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4426 0.0404 -0.0674
C -0.0099 0.0037 0.0091
H 1.6948 0.1792 0.8694
H -0.3694 -0.3717 -0.9490
H -0.2430 -0.7081 0.8010
H -0.4035 0.9993 0.2141

