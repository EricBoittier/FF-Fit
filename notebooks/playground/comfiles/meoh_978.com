%nproc=4
%mem=5760MB
%chk=meoh_978.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4482 0.0226 0.0515
C -0.0069 -0.0074 -0.0062
H 1.5626 0.5789 -0.7472
H -0.3745 0.9035 0.4664
H -0.2697 -0.1079 -1.0592
H -0.3649 -0.8593 0.5719

