%nproc=4
%mem=5760MB
%chk=meoh_613.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4341 0.0213 -0.0669
C -0.0007 -0.0055 0.0189
H 1.7110 0.5160 0.7325
H -0.3023 -1.0437 0.1578
H -0.3470 0.6213 0.8407
H -0.3574 0.4193 -0.9194

