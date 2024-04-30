%nproc=4
%mem=5760MB
%chk=meoh_149.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4307 0.0299 0.0512
C -0.0061 -0.0089 0.0007
H 1.7627 0.4106 -0.7886
H -0.3150 -1.0008 -0.3293
H -0.3670 0.2436 0.9977
H -0.2584 0.7626 -0.7268

