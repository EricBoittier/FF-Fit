%nproc=4
%mem=5760MB
%chk=meoh_680.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4373 0.0060 0.0300
C 0.0041 -0.0028 0.0071
H 1.6846 0.7261 -0.5870
H -0.3405 -0.9826 -0.3237
H -0.3978 0.2078 0.9982
H -0.3495 0.7716 -0.6736

