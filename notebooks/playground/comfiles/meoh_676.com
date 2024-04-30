%nproc=4
%mem=5760MB
%chk=meoh_676.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4380 0.0021 0.0224
C -0.0111 0.0020 0.0155
H 1.7153 0.7707 -0.5189
H -0.2747 -0.9971 -0.3315
H -0.3713 0.2187 1.0212
H -0.3034 0.7352 -0.7361

