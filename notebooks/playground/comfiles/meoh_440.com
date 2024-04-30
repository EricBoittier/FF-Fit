%nproc=4
%mem=5760MB
%chk=meoh_440.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4275 0.1018 -0.0405
C 0.0097 -0.0025 0.0055
H 1.7317 -0.6633 0.4911
H -0.3906 0.4681 0.9035
H -0.4343 0.4621 -0.8748
H -0.2215 -1.0674 0.0327

