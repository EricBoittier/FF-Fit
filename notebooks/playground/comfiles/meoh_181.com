%nproc=4
%mem=5760MB
%chk=meoh_181.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4233 0.0824 0.0532
C 0.0269 -0.0141 -0.0020
H 1.7083 -0.3295 -0.7892
H -0.3482 -0.8959 -0.5215
H -0.4445 0.0078 0.9805
H -0.3694 0.8626 -0.5143

