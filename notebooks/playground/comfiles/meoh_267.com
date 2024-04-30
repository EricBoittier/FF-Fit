%nproc=4
%mem=5760MB
%chk=meoh_267.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4208 0.0346 -0.0548
C 0.0147 -0.0016 -0.0020
H 1.8173 0.3186 0.7953
H -0.4229 -0.3257 -0.9463
H -0.2905 -0.7202 0.7586
H -0.3715 0.9829 0.2619

