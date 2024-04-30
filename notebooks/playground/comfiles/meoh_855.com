%nproc=4
%mem=5760MB
%chk=meoh_855.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4408 -0.0083 -0.0191
C 0.0016 0.0023 -0.0057
H 1.6654 0.9175 0.2113
H -0.4563 0.0873 -0.9912
H -0.3315 -0.9222 0.4660
H -0.3083 0.8069 0.6610

