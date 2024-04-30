%nproc=4
%mem=5760MB
%chk=meoh_771.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4186 0.1096 -0.0060
C 0.0235 -0.0003 0.0190
H 1.8077 -0.7840 -0.1082
H -0.2773 -0.6427 -0.8087
H -0.3616 -0.4570 0.9307
H -0.5073 0.9327 -0.1703

