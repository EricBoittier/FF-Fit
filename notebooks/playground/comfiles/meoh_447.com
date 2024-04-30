%nproc=4
%mem=5760MB
%chk=meoh_447.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4023 0.0751 -0.0474
C 0.0370 -0.0056 -0.0034
H 1.9042 -0.2886 0.7117
H -0.2966 0.3947 0.9540
H -0.4668 0.5603 -0.7870
H -0.3802 -1.0069 -0.1107

