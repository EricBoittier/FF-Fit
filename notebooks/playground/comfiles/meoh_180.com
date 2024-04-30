%nproc=4
%mem=5760MB
%chk=meoh_180.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4251 0.0814 0.0531
C 0.0214 -0.0151 -0.0009
H 1.7149 -0.3067 -0.7989
H -0.3404 -0.9033 -0.5189
H -0.4428 0.0170 0.9848
H -0.3478 0.8667 -0.5244

