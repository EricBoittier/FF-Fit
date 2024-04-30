%nproc=4
%mem=5760MB
%chk=meoh_375.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4201 0.0216 -0.0545
C 0.0104 -0.0062 0.0057
H 1.7781 0.5010 0.7218
H -0.3031 1.0201 -0.1855
H -0.3588 -0.6883 -0.7601
H -0.3220 -0.3175 0.9960

