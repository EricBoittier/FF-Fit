%nproc=4
%mem=5760MB
%chk=meoh_603.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4231 0.0336 -0.0688
C 0.0165 0.0087 0.0158
H 1.7699 0.2862 0.8124
H -0.2726 -1.0267 0.1960
H -0.3921 0.6264 0.8156
H -0.4301 0.2616 -0.9458

