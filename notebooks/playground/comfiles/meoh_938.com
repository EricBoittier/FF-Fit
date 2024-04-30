%nproc=4
%mem=5760MB
%chk=meoh_938.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4387 0.0910 -0.0666
C -0.0167 -0.0273 0.0110
H 1.6941 -0.2681 0.8089
H -0.3236 0.9875 -0.2422
H -0.2888 -0.7943 -0.7142
H -0.2595 -0.2582 1.0482

