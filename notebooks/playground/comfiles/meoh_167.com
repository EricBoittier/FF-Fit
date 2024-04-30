%nproc=4
%mem=5760MB
%chk=meoh_167.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4285 0.0525 0.0561
C 0.0066 0.0019 -0.0001
H 1.7949 0.0104 -0.8519
H -0.2900 -0.9525 -0.4352
H -0.4095 0.0753 1.0046
H -0.3899 0.7966 -0.6320

