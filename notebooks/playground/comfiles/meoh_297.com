%nproc=4
%mem=5760MB
%chk=meoh_297.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4187 -0.0017 0.0111
C 0.0137 0.0026 -0.0042
H 1.8546 0.8567 -0.1722
H -0.3845 0.0358 -1.0183
H -0.3236 -0.9351 0.4374
H -0.3692 0.8233 0.6024

