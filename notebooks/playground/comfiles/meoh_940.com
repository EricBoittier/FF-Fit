%nproc=4
%mem=5760MB
%chk=meoh_940.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4302 0.0820 -0.0644
C -0.0119 -0.0252 0.0103
H 1.7707 -0.1475 0.8255
H -0.3075 0.9929 -0.2432
H -0.3107 -0.7529 -0.7441
H -0.2529 -0.3076 1.0352

