%nproc=4
%mem=5760MB
%chk=meoh_115.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4263 -0.0062 0.0130
C 0.0195 -0.0030 0.0060
H 1.7211 0.8879 -0.2591
H -0.4054 -1.0009 -0.1027
H -0.3994 0.4517 0.9037
H -0.3293 0.5815 -0.8453

