%nproc=4
%mem=5760MB
%chk=meoh_904.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4173 0.0981 0.0363
C 0.0423 -0.0161 0.0154
H 1.7069 -0.4784 -0.7014
H -0.3861 0.6312 -0.7499
H -0.3559 -0.9954 -0.2498
H -0.5057 0.2614 0.9159

