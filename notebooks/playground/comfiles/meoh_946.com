%nproc=4
%mem=5760MB
%chk=meoh_946.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4069 0.0493 -0.0538
C 0.0291 -0.0128 0.0064
H 1.8899 0.2365 0.7781
H -0.3898 0.9734 -0.1939
H -0.4124 -0.6193 -0.7844
H -0.3067 -0.4353 0.9534

