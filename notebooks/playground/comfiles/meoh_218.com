%nproc=4
%mem=5760MB
%chk=meoh_218.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4273 0.1092 -0.0028
C -0.0107 -0.0057 0.0000
H 1.8268 -0.7856 0.0137
H -0.2559 -0.7557 -0.7521
H -0.2581 -0.3265 1.0119
H -0.3815 0.9872 -0.2544

