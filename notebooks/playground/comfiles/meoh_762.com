%nproc=4
%mem=5760MB
%chk=meoh_762.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4409 0.1157 0.0179
C -0.0129 -0.0149 -0.0042
H 1.7343 -0.7550 -0.3232
H -0.3207 -0.7368 -0.7606
H -0.2539 -0.3652 0.9995
H -0.4186 0.9823 -0.1749

