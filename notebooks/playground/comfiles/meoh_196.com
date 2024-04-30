%nproc=4
%mem=5760MB
%chk=meoh_196.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4339 0.0912 0.0355
C -0.0164 0.0062 -0.0033
H 1.8268 -0.6015 -0.5357
H -0.2006 -0.8848 -0.6036
H -0.3070 -0.1661 1.0330
H -0.4250 0.9162 -0.4425

