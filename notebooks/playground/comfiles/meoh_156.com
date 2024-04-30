%nproc=4
%mem=5760MB
%chk=meoh_156.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4327 0.0337 0.0633
C 0.0084 0.0033 -0.0148
H 1.6927 0.2642 -0.8531
H -0.3042 -0.9925 -0.3289
H -0.3713 0.1631 0.9944
H -0.3989 0.7758 -0.6670

