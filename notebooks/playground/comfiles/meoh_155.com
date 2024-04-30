%nproc=4
%mem=5760MB
%chk=meoh_155.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4338 0.0331 0.0623
C 0.0033 0.0014 -0.0136
H 1.6972 0.2862 -0.8471
H -0.2992 -0.9979 -0.3266
H -0.3627 0.1751 0.9983
H -0.3737 0.7799 -0.6768

