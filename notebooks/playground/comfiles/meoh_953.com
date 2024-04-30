%nproc=4
%mem=5760MB
%chk=meoh_953.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4376 0.0086 -0.0540
C 0.0157 0.0056 0.0091
H 1.6413 0.6512 0.6574
H -0.4576 0.9875 0.0173
H -0.3805 -0.5428 -0.8454
H -0.3507 -0.5145 0.8942

