%nproc=4
%mem=5760MB
%chk=meoh_862.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4210 -0.0082 -0.0011
C 0.0364 -0.0010 -0.0037
H 1.7074 0.9274 -0.0568
H -0.4889 0.2028 -0.9368
H -0.3510 -0.9488 0.3701
H -0.3975 0.7459 0.6610

