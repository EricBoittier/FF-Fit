%nproc=4
%mem=5760MB
%chk=meoh_560.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4321 0.1159 -0.0353
C 0.0113 -0.0217 0.0058
H 1.7399 -0.7260 0.3609
H -0.3203 -0.8925 0.5713
H -0.3800 0.8577 0.5174
H -0.4464 -0.0355 -0.9834

