%nproc=4
%mem=5760MB
%chk=meoh_328.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4448 0.0741 0.0553
C -0.0174 0.0025 -0.0085
H 1.6963 -0.3027 -0.8137
H -0.3635 0.4983 -0.9155
H -0.2002 -1.0717 0.0204
H -0.3995 0.4547 0.9067

