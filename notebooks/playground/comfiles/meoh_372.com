%nproc=4
%mem=5760MB
%chk=meoh_372.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4294 0.0295 -0.0592
C -0.0172 0.0038 0.0067
H 1.8040 0.3189 0.7989
H -0.3442 1.0089 -0.2597
H -0.2574 -0.7566 -0.7364
H -0.2271 -0.3001 1.0322

