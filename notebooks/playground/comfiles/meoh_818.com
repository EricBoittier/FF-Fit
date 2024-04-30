%nproc=4
%mem=5760MB
%chk=meoh_818.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4164 0.0632 -0.0629
C 0.0321 -0.0104 0.0029
H 1.7982 -0.0615 0.8311
H -0.4880 -0.2643 -0.9208
H -0.3088 -0.7188 0.7579
H -0.4064 0.9510 0.2705

