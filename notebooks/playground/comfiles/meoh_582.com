%nproc=4
%mem=5760MB
%chk=meoh_582.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4255 0.0797 -0.0681
C 0.0148 -0.0112 0.0160
H 1.7465 -0.2651 0.7913
H -0.3155 -0.9866 0.3732
H -0.3663 0.7666 0.6778
H -0.4080 0.1380 -0.9775

