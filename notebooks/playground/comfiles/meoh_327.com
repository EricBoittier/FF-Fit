%nproc=4
%mem=5760MB
%chk=meoh_327.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4440 0.0712 0.0547
C -0.0219 0.0009 -0.0059
H 1.7223 -0.2503 -0.8283
H -0.3339 0.4887 -0.9293
H -0.2042 -1.0732 0.0296
H -0.3845 0.4790 0.9041

