%nproc=4
%mem=5760MB
%chk=meoh_383.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4280 -0.0016 -0.0341
C 0.0368 -0.0189 0.0051
H 1.5938 0.8717 0.3787
H -0.2786 1.0243 -0.0151
H -0.5206 -0.4648 -0.8186
H -0.4408 -0.3961 0.9094

