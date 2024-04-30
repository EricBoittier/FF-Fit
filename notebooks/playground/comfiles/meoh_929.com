%nproc=4
%mem=5760MB
%chk=meoh_929.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4236 0.1064 -0.0445
C 0.0348 -0.0058 0.0028
H 1.6973 -0.6529 0.5115
H -0.5597 0.8350 -0.3546
H -0.3505 -0.8768 -0.5273
H -0.3386 -0.1396 1.0181

