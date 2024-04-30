%nproc=4
%mem=5760MB
%chk=meoh_188.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4238 0.0854 0.0493
C 0.0190 0.0002 -0.0071
H 1.7430 -0.4706 -0.6920
H -0.2924 -0.8861 -0.5599
H -0.3661 -0.0779 1.0096
H -0.4515 0.8623 -0.4799

