%nproc=4
%mem=5760MB
%chk=meoh_435.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4534 0.1154 -0.0313
C -0.0204 -0.0057 0.0142
H 1.5837 -0.8148 0.2486
H -0.4346 0.5324 0.8669
H -0.3793 0.3878 -0.9368
H -0.1380 -1.0838 0.1233

