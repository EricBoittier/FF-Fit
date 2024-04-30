%nproc=4
%mem=5760MB
%chk=meoh_570.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4288 0.0926 -0.0521
C -0.0027 0.0018 0.0128
H 1.8016 -0.5493 0.5879
H -0.1835 -0.9718 0.4685
H -0.4349 0.8078 0.6058
H -0.3719 0.0079 -1.0127

