%nproc=4
%mem=5760MB
%chk=meoh_457.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4556 0.0421 -0.0779
C -0.0217 -0.0104 0.0168
H 1.5885 0.2296 0.8749
H -0.2836 0.2617 1.0393
H -0.3463 0.7706 -0.6708
H -0.3442 -1.0213 -0.2326

