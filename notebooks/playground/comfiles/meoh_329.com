%nproc=4
%mem=5760MB
%chk=meoh_329.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4444 0.0770 0.0554
C -0.0108 0.0041 -0.0106
H 1.6736 -0.3533 -0.7948
H -0.3946 0.5058 -0.8989
H -0.2017 -1.0688 0.0087
H -0.4166 0.4307 0.9067

