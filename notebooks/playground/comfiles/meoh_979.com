%nproc=4
%mem=5760MB
%chk=meoh_979.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4537 0.0245 0.0537
C -0.0131 -0.0044 -0.0060
H 1.5349 0.5283 -0.7830
H -0.3864 0.8918 0.4898
H -0.2457 -0.0998 -1.0666
H -0.3629 -0.8720 0.5535

