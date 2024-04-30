%nproc=4
%mem=5760MB
%chk=meoh_504.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4367 0.0296 0.0515
C -0.0042 0.0098 0.0011
H 1.6863 0.3266 -0.8486
H -0.3367 -0.4829 0.9148
H -0.4155 1.0148 -0.0934
H -0.2294 -0.6606 -0.8283

