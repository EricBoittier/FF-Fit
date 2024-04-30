%nproc=4
%mem=5760MB
%chk=meoh_733.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4260 0.0802 0.0459
C 0.0223 -0.0046 0.0016
H 1.7236 -0.3153 -0.7999
H -0.2980 -0.8556 -0.5995
H -0.4142 -0.1705 0.9864
H -0.4531 0.9072 -0.3600

