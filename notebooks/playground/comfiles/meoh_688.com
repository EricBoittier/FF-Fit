%nproc=4
%mem=5760MB
%chk=meoh_688.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4281 0.0182 0.0416
C 0.0241 -0.0106 -0.0053
H 1.7117 0.6041 -0.6910
H -0.4279 -0.9463 -0.3344
H -0.3772 0.1604 0.9937
H -0.4024 0.8044 -0.5901

