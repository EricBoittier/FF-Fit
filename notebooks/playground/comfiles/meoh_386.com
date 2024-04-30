%nproc=4
%mem=5760MB
%chk=meoh_386.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4417 -0.0085 -0.0227
C 0.0092 -0.0149 0.0054
H 1.5702 0.9368 0.2019
H -0.2649 1.0396 0.0378
H -0.4653 -0.4153 -0.8905
H -0.3754 -0.4634 0.9214

