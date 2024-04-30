%nproc=4
%mem=5760MB
%chk=meoh_846.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4310 0.0079 -0.0467
C -0.0034 -0.0081 0.0184
H 1.7637 0.7568 0.4908
H -0.3060 0.0098 -1.0287
H -0.3541 -0.9145 0.5118
H -0.3192 0.9049 0.5230

