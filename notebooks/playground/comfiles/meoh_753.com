%nproc=4
%mem=5760MB
%chk=meoh_753.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4156 0.1024 0.0250
C 0.0283 -0.0119 0.0003
H 1.8254 -0.6211 -0.4938
H -0.3597 -0.7546 -0.6968
H -0.3737 -0.2717 0.9795
H -0.4405 0.9486 -0.2136

