%nproc=4
%mem=5760MB
%chk=meoh_832.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4369 0.0297 -0.0682
C -0.0008 0.0009 0.0131
H 1.6922 0.3773 0.8118
H -0.3929 -0.1784 -0.9880
H -0.2708 -0.8388 0.6534
H -0.3675 0.9417 0.4238

