%nproc=4
%mem=5760MB
%chk=meoh_851.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4461 -0.0023 -0.0323
C -0.0213 -0.0012 0.0041
H 1.6971 0.8656 0.3476
H -0.3558 0.0374 -1.0326
H -0.3187 -0.9272 0.4962
H -0.2638 0.8593 0.6277

