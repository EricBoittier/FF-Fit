%nproc=4
%mem=5760MB
%chk=meoh_610.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4373 0.0225 -0.0715
C -0.0087 0.0018 0.0240
H 1.6987 0.4529 0.7693
H -0.2477 -1.0530 0.1595
H -0.3613 0.6323 0.8402
H -0.3413 0.3733 -0.9453

