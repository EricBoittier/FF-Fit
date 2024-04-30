%nproc=4
%mem=5760MB
%chk=meoh_944.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4112 0.0607 -0.0564
C 0.0156 -0.0175 0.0074
H 1.8846 0.1088 0.8004
H -0.3484 0.9832 -0.2255
H -0.3823 -0.6612 -0.7771
H -0.2800 -0.4007 0.9841

