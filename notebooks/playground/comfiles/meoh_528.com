%nproc=4
%mem=5760MB
%chk=meoh_528.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4250 0.0987 0.0331
C 0.0105 -0.0116 0.0024
H 1.7921 -0.5328 -0.6203
H -0.3396 -0.6932 0.7775
H -0.3965 0.9761 0.2191
H -0.3417 -0.3944 -0.9555

