%nproc=4
%mem=5760MB
%chk=meoh_346.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4259 0.1113 -0.0088
C -0.0193 -0.0154 0.0112
H 1.8517 -0.7712 0.0096
H -0.2489 0.7636 -0.7158
H -0.2580 -1.0147 -0.3529
H -0.2883 0.2246 1.0399

