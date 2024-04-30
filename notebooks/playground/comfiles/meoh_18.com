%nproc=4
%mem=5760MB
%chk=meoh_18.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4309 -0.0104 -0.0004
C 0.0090 0.0016 0.0001
H 1.7213 0.9256 0.0004
H -0.3672 -1.0214 0.0000
H -0.3586 0.5144 0.8889
H -0.3587 0.5140 -0.8889

