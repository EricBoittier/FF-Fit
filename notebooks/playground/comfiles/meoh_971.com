%nproc=4
%mem=5760MB
%chk=meoh_971.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4035 0.0174 0.0222
C 0.0402 -0.0274 -0.0005
H 1.8324 0.8128 -0.3572
H -0.3223 0.9631 0.2747
H -0.4398 -0.1662 -0.9692
H -0.3685 -0.7739 0.6805

