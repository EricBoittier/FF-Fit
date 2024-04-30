%nproc=4
%mem=5760MB
%chk=meoh_357.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4243 0.0937 -0.0531
C 0.0387 -0.0042 0.0045
H 1.6831 -0.5562 0.6333
H -0.5160 0.8172 -0.4491
H -0.4001 -0.9008 -0.4332
H -0.3768 -0.0131 1.0122

