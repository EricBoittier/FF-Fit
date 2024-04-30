%nproc=4
%mem=5760MB
%chk=meoh_564.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4425 0.1096 -0.0437
C -0.0141 -0.0165 0.0089
H 1.7292 -0.6762 0.4671
H -0.2207 -0.9382 0.5529
H -0.3817 0.8530 0.5538
H -0.3973 0.0038 -1.0113

