%nproc=4
%mem=5760MB
%chk=meoh_963.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4394 -0.0027 -0.0186
C -0.0061 -0.0081 0.0030
H 1.6667 0.9345 0.1558
H -0.3095 1.0182 0.2101
H -0.3696 -0.3702 -0.9586
H -0.3039 -0.6576 0.8261

