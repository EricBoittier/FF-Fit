%nproc=4
%mem=5760MB
%chk=meoh_392.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4462 -0.0105 0.0011
C -0.0267 -0.0000 0.0079
H 1.6964 0.9197 -0.1797
H -0.3058 1.0436 0.1528
H -0.3096 -0.3789 -0.9742
H -0.2585 -0.6325 0.8648

