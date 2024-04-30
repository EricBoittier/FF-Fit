%nproc=4
%mem=5760MB
%chk=meoh_182.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4219 0.0832 0.0531
C 0.0311 -0.0126 -0.0032
H 1.7045 -0.3516 -0.7785
H -0.3525 -0.8895 -0.5247
H -0.4430 -0.0022 0.9782
H -0.3900 0.8583 -0.5054

