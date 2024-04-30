%nproc=4
%mem=5760MB
%chk=meoh_558.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4242 0.1167 -0.0307
C 0.0248 -0.0214 0.0049
H 1.7654 -0.7400 0.3014
H -0.3620 -0.8679 0.5723
H -0.3849 0.8590 0.5000
H -0.4603 -0.0623 -0.9704

