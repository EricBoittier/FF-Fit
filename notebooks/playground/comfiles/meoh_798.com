%nproc=4
%mem=5760MB
%chk=meoh_798.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4203 0.0959 -0.0465
C 0.0219 -0.0027 0.0037
H 1.8179 -0.5656 0.5575
H -0.4010 -0.4865 -0.8768
H -0.2787 -0.6075 0.8593
H -0.4830 0.9551 0.1289

