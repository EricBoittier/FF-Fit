%nproc=4
%mem=5760MB
%chk=meoh_257.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4460 0.0530 -0.0733
C -0.0040 0.0046 0.0151
H 1.6316 -0.0293 0.8855
H -0.3541 -0.4218 -0.9249
H -0.2688 -0.6486 0.8465
H -0.4533 0.9884 0.1513

