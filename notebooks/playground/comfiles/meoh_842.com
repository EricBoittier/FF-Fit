%nproc=4
%mem=5760MB
%chk=meoh_842.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4157 0.0146 -0.0552
C 0.0286 -0.0090 0.0252
H 1.7828 0.6527 0.5917
H -0.3394 -0.0138 -1.0008
H -0.3911 -0.8823 0.5245
H -0.4043 0.9034 0.4354

