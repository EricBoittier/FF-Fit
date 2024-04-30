%nproc=4
%mem=5760MB
%chk=meoh_273.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4106 0.0226 -0.0449
C 0.0302 -0.0092 -0.0055
H 1.8332 0.5321 0.6779
H -0.4422 -0.2288 -0.9630
H -0.3561 -0.7340 0.7110
H -0.3262 0.9657 0.3271

