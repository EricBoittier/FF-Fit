%nproc=4
%mem=5760MB
%chk=meoh_462.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4399 0.0229 -0.0706
C 0.0099 -0.0005 0.0164
H 1.6272 0.4711 0.7806
H -0.3642 0.1449 1.0298
H -0.4146 0.7850 -0.6088
H -0.3627 -0.9742 -0.3016

