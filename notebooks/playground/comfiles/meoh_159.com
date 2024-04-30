%nproc=4
%mem=5760MB
%chk=meoh_159.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4283 0.0366 0.0644
C 0.0226 0.0076 -0.0151
H 1.6994 0.1966 -0.8638
H -0.3212 -0.9729 -0.3443
H -0.4006 0.1299 0.9819
H -0.4590 0.7592 -0.6406

