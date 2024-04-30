%nproc=4
%mem=5760MB
%chk=meoh_520.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4320 0.0817 0.0444
C -0.0161 -0.0210 0.0065
H 1.8178 -0.2720 -0.7842
H -0.3082 -0.6149 0.8726
H -0.3023 1.0280 0.0825
H -0.2852 -0.4027 -0.9783

