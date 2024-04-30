%nproc=4
%mem=5760MB
%chk=meoh_379.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4159 0.0103 -0.0462
C 0.0429 -0.0168 0.0052
H 1.6886 0.7160 0.5768
H -0.2896 1.0167 -0.0921
H -0.4911 -0.5685 -0.7684
H -0.4348 -0.3419 0.9294

