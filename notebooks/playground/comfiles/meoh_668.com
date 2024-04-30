%nproc=4
%mem=5760MB
%chk=meoh_668.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4210 -0.0018 0.0119
C 0.0201 0.0028 0.0132
H 1.7540 0.8394 -0.3648
H -0.3254 -0.9959 -0.2543
H -0.3964 0.2495 0.9898
H -0.3670 0.6879 -0.7411

