%nproc=4
%mem=5760MB
%chk=meoh_423.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4046 0.0948 0.0179
C 0.0370 -0.0003 0.0091
H 1.9077 -0.6107 -0.4399
H -0.4170 0.7145 0.6955
H -0.4086 0.1535 -0.9737
H -0.3592 -0.9734 0.2994

