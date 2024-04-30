%nproc=4
%mem=5760MB
%chk=meoh_108.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4473 -0.0149 0.0038
C -0.0083 0.0029 0.0064
H 1.6137 0.9424 -0.1241
H -0.3601 -1.0270 -0.0548
H -0.3769 0.5071 0.8997
H -0.2916 0.5656 -0.8830

