%nproc=4
%mem=5760MB
%chk=meoh_703.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4267 0.0331 0.0451
C 0.0129 0.0009 0.0123
H 1.7265 0.3260 -0.8408
H -0.2638 -0.9359 -0.4714
H -0.4281 0.0323 1.0086
H -0.3758 0.8264 -0.5839

