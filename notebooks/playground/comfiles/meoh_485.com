%nproc=4
%mem=5760MB
%chk=meoh_485.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4385 -0.0091 0.0113
C 0.0200 -0.0054 0.0004
H 1.5820 0.9221 -0.2586
H -0.4240 -0.1457 0.9860
H -0.3283 0.9810 -0.3060
H -0.4418 -0.7640 -0.6314

