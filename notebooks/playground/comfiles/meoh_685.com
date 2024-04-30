%nproc=4
%mem=5760MB
%chk=meoh_685.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4299 0.0133 0.0382
C 0.0262 -0.0089 -0.0029
H 1.6883 0.6542 -0.6568
H -0.4243 -0.9505 -0.3170
H -0.4086 0.1821 0.9782
H -0.4057 0.7932 -0.6014

