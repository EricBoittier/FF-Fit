%nproc=4
%mem=5760MB
%chk=meoh_749.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4163 0.0966 0.0271
C 0.0237 -0.0082 0.0109
H 1.8267 -0.5643 -0.5691
H -0.3128 -0.7549 -0.7083
H -0.3887 -0.2655 0.9865
H -0.4305 0.9345 -0.2943

