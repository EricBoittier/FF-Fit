%nproc=4
%mem=5760MB
%chk=meoh_420.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4125 0.0899 0.0328
C 0.0206 -0.0100 0.0037
H 1.8746 -0.4927 -0.6055
H -0.3551 0.7877 0.6446
H -0.3665 0.1434 -1.0037
H -0.3591 -0.9620 0.3744

