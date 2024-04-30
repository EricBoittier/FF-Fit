%nproc=4
%mem=5760MB
%chk=meoh_959.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4600 -0.0076 -0.0385
C -0.0216 0.0060 0.0062
H 1.5301 0.8719 0.3882
H -0.3626 1.0269 0.1783
H -0.3196 -0.4725 -0.9267
H -0.3055 -0.5918 0.8723

