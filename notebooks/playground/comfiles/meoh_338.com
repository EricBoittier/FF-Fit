%nproc=4
%mem=5760MB
%chk=meoh_338.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4181 0.0984 0.0307
C 0.0378 0.0066 -0.0012
H 1.7314 -0.6913 -0.4578
H -0.4818 0.5702 -0.7762
H -0.2874 -1.0161 -0.1921
H -0.4620 0.2820 0.9275

