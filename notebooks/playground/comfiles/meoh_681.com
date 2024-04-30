%nproc=4
%mem=5760MB
%chk=meoh_681.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4359 0.0073 0.0319
C 0.0097 -0.0041 0.0047
H 1.6810 0.7131 -0.6024
H -0.3611 -0.9762 -0.3203
H -0.4046 0.2038 0.9912
H -0.3642 0.7779 -0.6561

