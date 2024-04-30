%nproc=4
%mem=5760MB
%chk=meoh_932.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4359 0.1057 -0.0569
C 0.0155 -0.0179 0.0069
H 1.6172 -0.5571 0.6419
H -0.4881 0.9024 -0.2890
H -0.3204 -0.8661 -0.5894
H -0.3259 -0.1597 1.0323

