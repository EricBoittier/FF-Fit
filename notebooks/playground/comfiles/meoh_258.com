%nproc=4
%mem=5760MB
%chk=meoh_258.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4490 0.0506 -0.0739
C -0.0092 0.0050 0.0157
H 1.6221 0.0041 0.8897
H -0.3490 -0.4137 -0.9316
H -0.2567 -0.6616 0.8418
H -0.4470 0.9927 0.1607

