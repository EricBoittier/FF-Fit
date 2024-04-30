%nproc=4
%mem=5760MB
%chk=meoh_994.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4075 0.0831 0.0347
C 0.0299 -0.0130 0.0159
H 1.8964 -0.3523 -0.6946
H -0.4533 0.7526 0.6228
H -0.3766 0.2393 -0.9634
H -0.3046 -1.0187 0.2705

