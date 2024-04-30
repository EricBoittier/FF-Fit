%nproc=4
%mem=5760MB
%chk=meoh_527.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4268 0.0971 0.0345
C 0.0024 -0.0138 0.0027
H 1.8065 -0.5015 -0.6422
H -0.3284 -0.6849 0.7954
H -0.3672 0.9912 0.2060
H -0.3270 -0.3966 -0.9633

