%nproc=4
%mem=5760MB
%chk=meoh_190.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4275 0.0859 0.0465
C 0.0051 0.0040 -0.0069
H 1.7706 -0.5052 -0.6559
H -0.2529 -0.8915 -0.5723
H -0.3302 -0.1058 1.0244
H -0.4474 0.8772 -0.4768

