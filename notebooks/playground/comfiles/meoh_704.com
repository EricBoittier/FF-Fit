%nproc=4
%mem=5760MB
%chk=meoh_704.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4258 0.0342 0.0455
C 0.0185 0.0012 0.0128
H 1.7166 0.3064 -0.8499
H -0.2702 -0.9305 -0.4737
H -0.4508 0.0260 0.9962
H -0.3876 0.8263 -0.5725

