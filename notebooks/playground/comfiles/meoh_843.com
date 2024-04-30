%nproc=4
%mem=5760MB
%chk=meoh_843.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4186 0.0131 -0.0533
C 0.0214 -0.0094 0.0241
H 1.7828 0.6790 0.5668
H -0.3279 -0.0059 -1.0084
H -0.3845 -0.8909 0.5203
H -0.3847 0.9060 0.4544

