%nproc=4
%mem=5760MB
%chk=meoh_381.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4201 0.0042 -0.0406
C 0.0456 -0.0191 0.0051
H 1.6364 0.8031 0.4842
H -0.2861 1.0177 -0.0519
H -0.5221 -0.5115 -0.7844
H -0.4537 -0.3636 0.9107

