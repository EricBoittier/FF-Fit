%nproc=4
%mem=5760MB
%chk=meoh_121.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4172 0.0009 0.0240
C 0.0217 -0.0022 -0.0026
H 1.8089 0.8097 -0.3670
H -0.3754 -1.0100 -0.1240
H -0.3586 0.3955 0.9383
H -0.3690 0.6020 -0.8214

