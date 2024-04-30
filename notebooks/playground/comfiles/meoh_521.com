%nproc=4
%mem=5760MB
%chk=meoh_521.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4326 0.0844 0.0428
C -0.0198 -0.0215 0.0062
H 1.8276 -0.3055 -0.7649
H -0.2996 -0.6253 0.8695
H -0.2919 1.0298 0.0998
H -0.2807 -0.3979 -0.9829

