%nproc=4
%mem=5760MB
%chk=meoh_640.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4193 -0.0064 -0.0319
C 0.0278 0.0046 0.0198
H 1.7406 0.8813 0.2312
H -0.3279 -1.0224 -0.0625
H -0.4356 0.4272 0.9113
H -0.3775 0.5458 -0.8351

