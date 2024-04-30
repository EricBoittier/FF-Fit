%nproc=4
%mem=5760MB
%chk=meoh_710.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4287 0.0437 0.0497
C 0.0257 -0.0019 0.0104
H 1.6774 0.1850 -0.8877
H -0.3087 -0.9062 -0.4980
H -0.4886 -0.0106 0.9714
H -0.4045 0.8453 -0.5238

