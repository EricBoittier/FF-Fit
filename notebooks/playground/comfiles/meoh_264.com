%nproc=4
%mem=5760MB
%chk=meoh_264.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4376 0.0388 -0.0645
C -0.0049 0.0027 0.0064
H 1.7262 0.2146 0.8554
H -0.3818 -0.3617 -0.9493
H -0.2513 -0.7130 0.7907
H -0.3949 0.9966 0.2259

