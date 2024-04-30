%nproc=4
%mem=5760MB
%chk=meoh_230.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4288 0.1055 -0.0308
C 0.0241 0.0014 0.0030
H 1.7171 -0.7389 0.3745
H -0.3431 -0.6697 -0.7735
H -0.3404 -0.4252 0.9374
H -0.5405 0.9269 -0.1097

