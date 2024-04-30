%nproc=4
%mem=5760MB
%chk=meoh_354.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4265 0.1052 -0.0436
C 0.0183 -0.0153 0.0060
H 1.7243 -0.6635 0.4864
H -0.4018 0.8428 -0.5188
H -0.3723 -0.9481 -0.4007
H -0.3514 0.0661 1.0281

