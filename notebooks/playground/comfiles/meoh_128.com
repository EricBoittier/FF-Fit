%nproc=4
%mem=5760MB
%chk=meoh_128.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4386 -0.0018 0.0395
C -0.0088 0.0075 -0.0128
H 1.7159 0.7574 -0.5148
H -0.2913 -1.0364 -0.1489
H -0.3251 0.3456 0.9740
H -0.3711 0.6576 -0.8092

