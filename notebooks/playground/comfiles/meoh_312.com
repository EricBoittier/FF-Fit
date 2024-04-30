%nproc=4
%mem=5760MB
%chk=meoh_312.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4260 0.0184 0.0491
C 0.0327 -0.0082 0.0041
H 1.6356 0.5292 -0.7606
H -0.3268 0.3100 -0.9746
H -0.4517 -0.9641 0.2029
H -0.4224 0.7157 0.6801

