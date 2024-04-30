%nproc=4
%mem=5760MB
%chk=meoh_425.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4080 0.0997 0.0090
C 0.0365 0.0035 0.0118
H 1.8787 -0.6906 -0.3292
H -0.4483 0.6691 0.7260
H -0.4200 0.1728 -0.9634
H -0.3351 -0.9908 0.2593

