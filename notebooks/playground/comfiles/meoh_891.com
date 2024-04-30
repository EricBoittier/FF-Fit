%nproc=4
%mem=5760MB
%chk=meoh_891.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4371 0.0456 0.0557
C -0.0158 -0.0023 -0.0085
H 1.7701 0.1443 -0.8607
H -0.3909 0.5501 -0.8700
H -0.2257 -1.0719 -0.0057
H -0.3174 0.4661 0.9283

