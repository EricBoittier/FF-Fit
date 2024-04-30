%nproc=4
%mem=5760MB
%chk=meoh_179.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4271 0.0801 0.0529
C 0.0150 -0.0157 0.0003
H 1.7240 -0.2833 -0.8075
H -0.3299 -0.9113 -0.5167
H -0.4386 0.0252 0.9906
H -0.3266 0.8700 -0.5353

