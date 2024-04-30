%nproc=4
%mem=5760MB
%chk=meoh_275.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4185 0.0162 -0.0456
C 0.0217 -0.0090 -0.0015
H 1.7755 0.6083 0.6490
H -0.4197 -0.1927 -0.9811
H -0.3588 -0.7487 0.7029
H -0.3118 0.9688 0.3459

