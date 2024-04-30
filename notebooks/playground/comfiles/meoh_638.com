%nproc=4
%mem=5760MB
%chk=meoh_638.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4248 -0.0076 -0.0354
C 0.0186 0.0066 0.0192
H 1.7175 0.8734 0.2789
H -0.3001 -1.0340 -0.0417
H -0.4212 0.4442 0.9153
H -0.3746 0.5431 -0.8444

