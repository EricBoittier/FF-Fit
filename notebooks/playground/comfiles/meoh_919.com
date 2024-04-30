%nproc=4
%mem=5760MB
%chk=meoh_919.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4274 0.1088 -0.0037
C -0.0154 -0.0014 0.0047
H 1.8747 -0.7626 -0.0377
H -0.4084 0.8064 -0.6126
H -0.2199 -0.9757 -0.4393
H -0.2611 0.0072 1.0666

