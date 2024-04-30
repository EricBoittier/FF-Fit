%nproc=4
%mem=5760MB
%chk=meoh_869.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4315 0.0014 0.0101
C -0.0100 -0.0067 0.0146
H 1.7410 0.8644 -0.3361
H -0.2435 0.2708 -1.0133
H -0.3136 -1.0281 0.2443
H -0.3288 0.7361 0.7458

