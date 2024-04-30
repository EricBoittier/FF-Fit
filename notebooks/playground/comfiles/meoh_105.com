%nproc=4
%mem=5760MB
%chk=meoh_105.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4428 -0.0141 0.0009
C -0.0045 0.0030 0.0037
H 1.6491 0.9420 -0.0617
H -0.3570 -1.0280 -0.0280
H -0.3653 0.5147 0.8959
H -0.3160 0.5440 -0.8898

