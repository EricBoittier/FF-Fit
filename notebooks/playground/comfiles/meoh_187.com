%nproc=4
%mem=5760MB
%chk=meoh_187.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4223 0.0852 0.0504
C 0.0251 -0.0020 -0.0069
H 1.7306 -0.4524 -0.7088
H -0.3110 -0.8834 -0.5531
H -0.3847 -0.0639 1.0012
H -0.4498 0.8568 -0.4814

