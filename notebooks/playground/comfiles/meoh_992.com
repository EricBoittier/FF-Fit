%nproc=4
%mem=5760MB
%chk=meoh_992.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4073 0.0724 0.0370
C 0.0341 -0.0082 0.0216
H 1.8942 -0.2302 -0.7578
H -0.4920 0.7697 0.5748
H -0.3595 0.1948 -0.9743
H -0.3287 -1.0011 0.2873

