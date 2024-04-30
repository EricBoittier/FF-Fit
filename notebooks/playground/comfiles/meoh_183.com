%nproc=4
%mem=5760MB
%chk=meoh_183.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4209 0.0838 0.0529
C 0.0336 -0.0108 -0.0042
H 1.7038 -0.3732 -0.7666
H -0.3527 -0.8847 -0.5289
H -0.4380 -0.0131 0.9784
H -0.4085 0.8546 -0.4978

