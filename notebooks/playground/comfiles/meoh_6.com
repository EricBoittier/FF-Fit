%nproc=4
%mem=5760MB
%chk=meoh_6.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4247 -0.0092 -0.0002
C 0.0149 0.0023 0.0001
H 1.7238 0.9241 0.0002
H -0.3659 -1.0191 -0.0001
H -0.3603 0.5130 0.8869
H -0.3603 0.5127 -0.8869

