%nproc=4
%mem=5760MB
%chk=meoh_945.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4083 0.0550 -0.0549
C 0.0229 -0.0152 0.0068
H 1.8924 0.1728 0.7891
H -0.3686 0.9783 -0.2119
H -0.3989 -0.6394 -0.7810
H -0.2929 -0.4192 0.9686

