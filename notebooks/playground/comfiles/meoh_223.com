%nproc=4
%mem=5760MB
%chk=meoh_223.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4202 0.1020 -0.0133
C 0.0104 0.0084 -0.0021
H 1.8402 -0.7658 0.1626
H -0.2762 -0.7425 -0.7384
H -0.2825 -0.3856 0.9711
H -0.4888 0.9599 -0.1852

