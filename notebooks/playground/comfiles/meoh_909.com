%nproc=4
%mem=5760MB
%chk=meoh_909.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4274 0.1137 0.0249
C 0.0262 -0.0266 0.0143
H 1.6560 -0.6600 -0.5315
H -0.3609 0.7324 -0.6655
H -0.3082 -1.0053 -0.3301
H -0.4965 0.2309 0.9355

