%nproc=4
%mem=5760MB
%chk=meoh_758.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4315 0.1118 0.0231
C 0.0032 -0.0163 -0.0082
H 1.7666 -0.7040 -0.4041
H -0.3514 -0.7592 -0.7227
H -0.2986 -0.3103 0.9970
H -0.4171 0.9773 -0.1639

