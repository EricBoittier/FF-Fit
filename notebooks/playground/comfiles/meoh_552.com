%nproc=4
%mem=5760MB
%chk=meoh_552.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4136 0.1131 -0.0176
C 0.0256 -0.0143 0.0070
H 1.8453 -0.7575 0.1091
H -0.3490 -0.8446 0.6058
H -0.3860 0.8941 0.4469
H -0.3938 -0.1324 -0.9921

