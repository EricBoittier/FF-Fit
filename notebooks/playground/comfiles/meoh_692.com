%nproc=4
%mem=5760MB
%chk=meoh_692.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4303 0.0237 0.0439
C 0.0049 -0.0092 -0.0028
H 1.7476 0.5326 -0.7313
H -0.3734 -0.9590 -0.3809
H -0.3149 0.1247 1.0306
H -0.3619 0.8205 -0.6070

