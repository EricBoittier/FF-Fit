%nproc=4
%mem=5760MB
%chk=meoh_783.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4339 0.1181 -0.0265
C -0.0087 -0.0175 0.0103
H 1.7559 -0.7770 0.2094
H -0.2917 -0.6001 -0.8664
H -0.2891 -0.5129 0.9399
H -0.3729 1.0096 -0.0110

