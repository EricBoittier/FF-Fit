%nproc=4
%mem=5760MB
%chk=meoh_737.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4376 0.0864 0.0432
C -0.0001 -0.0046 0.0076
H 1.6855 -0.3995 -0.7710
H -0.2331 -0.8403 -0.6523
H -0.3935 -0.2164 1.0018
H -0.4174 0.9248 -0.3799

