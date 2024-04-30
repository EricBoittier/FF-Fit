%nproc=4
%mem=5760MB
%chk=meoh_579.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4140 0.0800 -0.0635
C 0.0314 -0.0005 0.0157
H 1.7935 -0.3332 0.7401
H -0.3097 -0.9724 0.3723
H -0.4269 0.7518 0.6577
H -0.4162 0.0743 -0.9753

