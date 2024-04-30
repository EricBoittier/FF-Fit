%nproc=4
%mem=5760MB
%chk=meoh_569.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4330 0.0954 -0.0511
C -0.0088 -0.0013 0.0124
H 1.7877 -0.5738 0.5709
H -0.1765 -0.9691 0.4849
H -0.4239 0.8187 0.5984
H -0.3693 0.0095 -1.0162

