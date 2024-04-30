%nproc=4
%mem=5760MB
%chk=meoh_643.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4170 -0.0042 -0.0256
C 0.0305 0.0001 0.0184
H 1.7639 0.8933 0.1605
H -0.3660 -1.0091 -0.0931
H -0.4198 0.4104 0.9223
H -0.3734 0.5553 -0.8282

