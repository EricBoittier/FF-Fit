%nproc=4
%mem=5760MB
%chk=meoh_720.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4348 0.0632 0.0519
C -0.0123 -0.0130 0.0015
H 1.7815 -0.0231 -0.8607
H -0.3200 -0.8860 -0.5742
H -0.2965 -0.0623 1.0526
H -0.3332 0.9082 -0.4848

