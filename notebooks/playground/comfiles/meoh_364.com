%nproc=4
%mem=5760MB
%chk=meoh_364.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4339 0.0589 -0.0641
C 0.0046 0.0172 0.0057
H 1.7158 -0.1887 0.8413
H -0.5515 0.8737 -0.3754
H -0.2759 -0.8526 -0.5883
H -0.2450 -0.1871 1.0469

