%nproc=4
%mem=5760MB
%chk=meoh_363.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4319 0.0638 -0.0635
C 0.0145 0.0160 0.0051
H 1.7018 -0.2487 0.8253
H -0.5683 0.8525 -0.3807
H -0.3027 -0.8578 -0.5640
H -0.2727 -0.1644 1.0410

