%nproc=4
%mem=5760MB
%chk=meoh_577.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4104 0.0806 -0.0602
C 0.0346 0.0051 0.0151
H 1.8189 -0.3780 0.7035
H -0.2911 -0.9688 0.3805
H -0.4536 0.7490 0.6447
H -0.4140 0.0417 -0.9776

