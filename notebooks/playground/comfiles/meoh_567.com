%nproc=4
%mem=5760MB
%chk=meoh_567.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4397 0.1014 -0.0486
C -0.0167 -0.0079 0.0112
H 1.7590 -0.6195 0.5335
H -0.1780 -0.9603 0.5163
H -0.4028 0.8369 0.5817
H -0.3723 0.0116 -1.0190

