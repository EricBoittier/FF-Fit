%nproc=4
%mem=5760MB
%chk=meoh_303.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4457 -0.0050 0.0361
C -0.0167 0.0054 -0.0117
H 1.6535 0.8127 -0.4626
H -0.3262 0.1616 -1.0451
H -0.2835 -0.9745 0.3841
H -0.3327 0.7995 0.6648

