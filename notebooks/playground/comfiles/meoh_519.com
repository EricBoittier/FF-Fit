%nproc=4
%mem=5760MB
%chk=meoh_519.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4309 0.0788 0.0460
C -0.0105 -0.0199 0.0066
H 1.8043 -0.2383 -0.8028
H -0.3208 -0.6044 0.8728
H -0.3186 1.0240 0.0665
H -0.2932 -0.4101 -0.9710

