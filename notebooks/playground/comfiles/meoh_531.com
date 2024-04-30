%nproc=4
%mem=5760MB
%chk=meoh_531.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4217 0.1029 0.0287
C 0.0310 -0.0049 0.0027
H 1.7433 -0.6199 -0.5498
H -0.3602 -0.7180 0.7283
H -0.4833 0.9248 0.2462
H -0.3759 -0.3765 -0.9377

