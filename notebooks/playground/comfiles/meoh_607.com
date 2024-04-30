%nproc=4
%mem=5760MB
%chk=meoh_607.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4339 0.0258 -0.0724
C -0.0036 0.0074 0.0239
H 1.7161 0.3845 0.7949
H -0.2338 -1.0485 0.1658
H -0.3771 0.6323 0.8352
H -0.3620 0.3192 -0.9571

