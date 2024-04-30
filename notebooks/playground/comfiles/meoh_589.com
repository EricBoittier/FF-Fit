%nproc=4
%mem=5760MB
%chk=meoh_589.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4426 0.0724 -0.0706
C -0.0142 -0.0241 0.0088
H 1.7189 -0.0877 0.8560
H -0.3042 -1.0089 0.3752
H -0.2675 0.7597 0.7227
H -0.4167 0.2592 -0.9637

