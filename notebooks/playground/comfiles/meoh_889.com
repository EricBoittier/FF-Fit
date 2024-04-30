%nproc=4
%mem=5760MB
%chk=meoh_889.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4339 0.0389 0.0558
C 0.0009 -0.0039 -0.0090
H 1.7353 0.2376 -0.8554
H -0.4195 0.5352 -0.8580
H -0.2698 -1.0593 0.0222
H -0.3589 0.5004 0.8879

