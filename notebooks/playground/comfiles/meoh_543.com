%nproc=4
%mem=5760MB
%chk=meoh_543.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4347 0.1105 0.0019
C -0.0158 -0.0046 0.0124
H 1.7652 -0.7917 -0.1909
H -0.1907 -0.8470 0.6818
H -0.4511 0.9404 0.3371
H -0.2480 -0.2153 -1.0315

