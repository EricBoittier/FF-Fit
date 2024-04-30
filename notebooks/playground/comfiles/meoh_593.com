%nproc=4
%mem=5760MB
%chk=meoh_593.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4325 0.0623 -0.0668
C 0.0030 -0.0168 0.0026
H 1.7703 0.0224 0.8523
H -0.3233 -1.0008 0.3395
H -0.2973 0.7131 0.7544
H -0.4647 0.2627 -0.9415

