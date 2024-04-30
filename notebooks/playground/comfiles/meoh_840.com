%nproc=4
%mem=5760MB
%chk=meoh_840.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4128 0.0173 -0.0589
C 0.0378 -0.0072 0.0262
H 1.7731 0.6002 0.6417
H -0.3620 -0.0352 -0.9875
H -0.3944 -0.8675 0.5374
H -0.4327 0.8993 0.4069

