%nproc=4
%mem=5760MB
%chk=meoh_802.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4294 0.0913 -0.0595
C -0.0000 -0.0023 0.0164
H 1.7863 -0.4912 0.6433
H -0.2892 -0.4744 -0.9225
H -0.2881 -0.6552 0.8403
H -0.4388 0.9847 0.1624

