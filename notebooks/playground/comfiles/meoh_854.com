%nproc=4
%mem=5760MB
%chk=meoh_854.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4435 -0.0071 -0.0224
C -0.0063 0.0018 -0.0038
H 1.6691 0.9078 0.2470
H -0.4307 0.0724 -1.0053
H -0.3255 -0.9236 0.4757
H -0.2912 0.8200 0.6576

