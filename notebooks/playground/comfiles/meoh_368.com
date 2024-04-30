%nproc=4
%mem=5760MB
%chk=meoh_368.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4373 0.0419 -0.0635
C -0.0252 0.0146 0.0074
H 1.7771 0.0638 0.8555
H -0.4462 0.9588 -0.3383
H -0.2086 -0.8168 -0.6732
H -0.1776 -0.2599 1.0512

