%nproc=4
%mem=5760MB
%chk=meoh_776.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4151 0.1104 -0.0181
C 0.0260 -0.0032 0.0219
H 1.8174 -0.7824 0.0217
H -0.2522 -0.6254 -0.8288
H -0.3962 -0.4757 0.9088
H -0.4807 0.9540 -0.1011

