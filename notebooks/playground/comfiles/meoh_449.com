%nproc=4
%mem=5760MB
%chk=meoh_449.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4103 0.0694 -0.0536
C 0.0250 -0.0094 -0.0007
H 1.8719 -0.1842 0.7729
H -0.2687 0.3784 0.9748
H -0.4403 0.6077 -0.7693
H -0.3883 -1.0078 -0.1436

