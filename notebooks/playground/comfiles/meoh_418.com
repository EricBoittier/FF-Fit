%nproc=4
%mem=5760MB
%chk=meoh_418.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4238 0.0869 0.0432
C 0.0037 -0.0166 -0.0007
H 1.8081 -0.4094 -0.7095
H -0.3181 0.8304 0.6052
H -0.3310 0.1397 -1.0262
H -0.3438 -0.9566 0.4279

