%nproc=4
%mem=5760MB
%chk=meoh_687.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4283 0.0166 0.0406
C 0.0263 -0.0103 -0.0049
H 1.7029 0.6213 -0.6802
H -0.4316 -0.9457 -0.3265
H -0.3907 0.1681 0.9863
H -0.4071 0.8002 -0.5908

