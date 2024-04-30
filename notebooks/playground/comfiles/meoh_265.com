%nproc=4
%mem=5760MB
%chk=meoh_265.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4320 0.0373 -0.0613
C 0.0012 0.0015 0.0035
H 1.7586 0.2497 0.8380
H -0.3955 -0.3508 -0.9487
H -0.2625 -0.7165 0.7801
H -0.3868 0.9927 0.2379

