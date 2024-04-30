%nproc=4
%mem=5760MB
%chk=meoh_347.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4275 0.1123 -0.0136
C -0.0220 -0.0177 0.0109
H 1.8458 -0.7700 0.0715
H -0.2391 0.7849 -0.6940
H -0.2635 -1.0130 -0.3621
H -0.2810 0.2122 1.0445

