%nproc=4
%mem=5760MB
%chk=meoh_489.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4235 0.0011 0.0202
C 0.0378 -0.0132 0.0047
H 1.6783 0.8403 -0.4173
H -0.4468 -0.1952 0.9640
H -0.3490 0.9694 -0.2653
H -0.4694 -0.6903 -0.6826

