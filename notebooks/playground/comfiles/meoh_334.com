%nproc=4
%mem=5760MB
%chk=meoh_334.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4300 0.0899 0.0467
C 0.0311 0.0097 -0.0106
H 1.6492 -0.5721 -0.6419
H -0.5109 0.5237 -0.8045
H -0.2600 -1.0378 -0.0878
H -0.4881 0.3281 0.8934

