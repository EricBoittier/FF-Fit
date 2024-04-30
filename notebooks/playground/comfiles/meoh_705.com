%nproc=4
%mem=5760MB
%chk=meoh_705.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4252 0.0354 0.0461
C 0.0232 0.0013 0.0130
H 1.7068 0.2866 -0.8584
H -0.2778 -0.9252 -0.4760
H -0.4703 0.0199 0.9847
H -0.3978 0.8266 -0.5612

