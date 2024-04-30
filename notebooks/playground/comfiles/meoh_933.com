%nproc=4
%mem=5760MB
%chk=meoh_933.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4393 0.1046 -0.0602
C 0.0073 -0.0212 0.0082
H 1.6063 -0.5171 0.6788
H -0.4565 0.9242 -0.2733
H -0.3080 -0.8603 -0.6119
H -0.3152 -0.1697 1.0387

