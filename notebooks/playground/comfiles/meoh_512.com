%nproc=4
%mem=5760MB
%chk=meoh_512.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4235 0.0546 0.0555
C 0.0384 -0.0000 0.0028
H 1.6701 0.0132 -0.8921
H -0.4317 -0.5320 0.8299
H -0.4794 0.9591 -0.0059
H -0.3526 -0.5216 -0.8708

