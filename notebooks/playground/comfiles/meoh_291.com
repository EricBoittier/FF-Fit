%nproc=4
%mem=5760MB
%chk=meoh_291.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4198 -0.0057 -0.0147
C 0.0272 0.0025 0.0068
H 1.7900 0.8923 0.1158
H -0.4017 -0.0326 -0.9947
H -0.3824 -0.8799 0.4984
H -0.4068 0.8671 0.5089

