%nproc=4
%mem=5760MB
%chk=meoh_899.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4254 0.0772 0.0453
C 0.0027 -0.0035 0.0079
H 1.7990 -0.2443 -0.8017
H -0.3487 0.5700 -0.8498
H -0.2735 -1.0435 -0.1660
H -0.3747 0.3186 0.9784

