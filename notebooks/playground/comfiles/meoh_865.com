%nproc=4
%mem=5760MB
%chk=meoh_865.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4223 -0.0046 0.0038
C 0.0212 -0.0042 0.0045
H 1.7332 0.9072 -0.1764
H -0.3965 0.2421 -0.9717
H -0.3308 -0.9864 0.3199
H -0.3761 0.7447 0.6895

