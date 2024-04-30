%nproc=4
%mem=5760MB
%chk=meoh_555.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4150 0.1155 -0.0239
C 0.0342 -0.0186 0.0051
H 1.8117 -0.7505 0.2069
H -0.3837 -0.8447 0.5804
H -0.3887 0.8697 0.4744
H -0.4480 -0.1011 -0.9690

