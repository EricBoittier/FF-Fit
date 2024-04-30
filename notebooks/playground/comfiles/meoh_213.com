%nproc=4
%mem=5760MB
%chk=meoh_213.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4292 0.1173 0.0063
C 0.0021 -0.0203 0.0029
H 1.7531 -0.7975 -0.1302
H -0.3128 -0.7499 -0.7433
H -0.3350 -0.2514 1.0133
H -0.3562 0.9638 -0.2992

