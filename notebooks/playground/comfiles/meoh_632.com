%nproc=4
%mem=5760MB
%chk=meoh_632.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4415 -0.0043 -0.0414
C -0.0107 0.0046 0.0116
H 1.6792 0.8306 0.4136
H -0.2833 -1.0508 0.0166
H -0.3106 0.4966 0.9369
H -0.3789 0.5230 -0.8737

