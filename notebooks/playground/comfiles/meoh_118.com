%nproc=4
%mem=5760MB
%chk=meoh_118.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4175 -0.0014 0.0180
C 0.0270 -0.0041 0.0025
H 1.7838 0.8458 -0.3116
H -0.4037 -0.9985 -0.1154
H -0.3865 0.4223 0.9164
H -0.3554 0.5869 -0.8297

