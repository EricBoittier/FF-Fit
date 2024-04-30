%nproc=4
%mem=5760MB
%chk=meoh_662.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4261 -0.0056 0.0088
C 0.0263 -0.0028 -0.0002
H 1.6976 0.8999 -0.2498
H -0.3960 -0.9935 -0.1683
H -0.3744 0.3055 0.9655
H -0.4179 0.6946 -0.7105

