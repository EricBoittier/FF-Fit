%nproc=4
%mem=5760MB
%chk=meoh_238.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4386 0.1042 -0.0441
C -0.0122 -0.0179 0.0068
H 1.7333 -0.5995 0.5711
H -0.3300 -0.5853 -0.8680
H -0.2763 -0.4837 0.9562
H -0.3582 1.0131 -0.0664

