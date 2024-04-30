%nproc=4
%mem=5760MB
%chk=meoh_198.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4325 0.0949 0.0316
C -0.0099 0.0033 -0.0016
H 1.8240 -0.6338 -0.4941
H -0.2218 -0.8718 -0.6160
H -0.3362 -0.1738 1.0232
H -0.4263 0.9194 -0.4205

