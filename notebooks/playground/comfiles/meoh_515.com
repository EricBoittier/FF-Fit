%nproc=4
%mem=5760MB
%chk=meoh_515.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4250 0.0655 0.0522
C 0.0220 -0.0100 0.0052
H 1.7275 -0.0984 -0.8654
H -0.3931 -0.5610 0.8491
H -0.4175 0.9874 0.0182
H -0.3391 -0.4637 -0.9178

