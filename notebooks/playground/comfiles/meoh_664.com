%nproc=4
%mem=5760MB
%chk=meoh_664.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4213 -0.0043 0.0098
C 0.0309 -0.0007 0.0039
H 1.7212 0.8798 -0.2883
H -0.3816 -0.9904 -0.1920
H -0.3951 0.2835 0.9661
H -0.4135 0.6895 -0.7131

