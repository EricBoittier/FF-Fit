%nproc=4
%mem=5760MB
%chk=meoh_769.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4252 0.1113 -0.0002
C 0.0129 -0.0027 0.0148
H 1.7868 -0.7857 -0.1587
H -0.2856 -0.6587 -0.8029
H -0.3253 -0.4469 0.9510
H -0.4928 0.9415 -0.1877

