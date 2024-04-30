%nproc=4
%mem=5760MB
%chk=meoh_212.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4284 0.1180 0.0080
C 0.0079 -0.0218 0.0032
H 1.7410 -0.7959 -0.1581
H -0.3282 -0.7497 -0.7352
H -0.3580 -0.2383 1.0069
H -0.3633 0.9556 -0.3049

