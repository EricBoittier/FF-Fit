%nproc=4
%mem=5760MB
%chk=meoh_642.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4168 -0.0049 -0.0278
C 0.0313 0.0017 0.0192
H 1.7586 0.8889 0.1839
H -0.3548 -1.0125 -0.0832
H -0.4301 0.4147 0.9162
H -0.3762 0.5509 -0.8296

