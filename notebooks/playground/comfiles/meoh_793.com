%nproc=4
%mem=5760MB
%chk=meoh_793.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4215 0.1055 -0.0341
C 0.0232 -0.0110 -0.0047
H 1.8040 -0.6562 0.4497
H -0.4716 -0.4969 -0.8457
H -0.2445 -0.5636 0.8959
H -0.4685 0.9587 0.0725

