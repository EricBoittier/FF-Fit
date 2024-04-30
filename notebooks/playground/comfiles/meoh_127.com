%nproc=4
%mem=5760MB
%chk=meoh_127.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4356 -0.0015 0.0375
C -0.0062 0.0064 -0.0121
H 1.7352 0.7655 -0.4940
H -0.2970 -1.0359 -0.1429
H -0.3214 0.3512 0.9727
H -0.3707 0.6507 -0.8123

