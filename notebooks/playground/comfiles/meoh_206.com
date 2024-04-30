%nproc=4
%mem=5760MB
%chk=meoh_206.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4236 0.1133 0.0179
C 0.0306 -0.0174 0.0038
H 1.7382 -0.7531 -0.3151
H -0.3642 -0.7773 -0.6707
H -0.4489 -0.1871 0.9678
H -0.4266 0.9121 -0.3353

