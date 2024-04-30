%nproc=4
%mem=5760MB
%chk=meoh_970.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4022 0.0163 0.0173
C 0.0405 -0.0279 0.0000
H 1.8465 0.8332 -0.2922
H -0.3163 0.9698 0.2558
H -0.4480 -0.1827 -0.9620
H -0.3631 -0.7617 0.6977

