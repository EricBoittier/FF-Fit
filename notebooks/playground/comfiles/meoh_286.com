%nproc=4
%mem=5760MB
%chk=meoh_286.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4449 -0.0106 -0.0345
C -0.0002 0.0037 0.0142
H 1.6031 0.8828 0.3361
H -0.3511 -0.0581 -1.0159
H -0.3621 -0.8515 0.5850
H -0.3621 0.9359 0.4480

