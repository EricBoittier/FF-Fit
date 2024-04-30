%nproc=4
%mem=5760MB
%chk=meoh_467.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4120 0.0104 -0.0481
C 0.0381 0.0048 0.0093
H 1.8009 0.6559 0.5785
H -0.3689 0.0448 1.0198
H -0.4714 0.7795 -0.5636
H -0.3680 -0.9169 -0.4074

