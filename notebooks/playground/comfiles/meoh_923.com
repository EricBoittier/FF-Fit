%nproc=4
%mem=5760MB
%chk=meoh_923.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4158 0.1041 -0.0174
C 0.0168 0.0076 0.0010
H 1.8787 -0.7354 0.1860
H -0.5206 0.7908 -0.5337
H -0.2970 -0.9286 -0.4605
H -0.2748 -0.0848 1.0471

