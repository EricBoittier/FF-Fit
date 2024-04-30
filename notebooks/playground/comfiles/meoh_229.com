%nproc=4
%mem=5760MB
%chk=meoh_229.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4260 0.1046 -0.0283
C 0.0267 0.0043 0.0018
H 1.7364 -0.7466 0.3453
H -0.3391 -0.6814 -0.7625
H -0.3405 -0.4197 0.9365
H -0.5494 0.9220 -0.1164

