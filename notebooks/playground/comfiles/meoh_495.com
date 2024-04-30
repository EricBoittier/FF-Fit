%nproc=4
%mem=5760MB
%chk=meoh_495.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4238 0.0157 0.0309
C 0.0039 -0.0120 0.0105
H 1.8069 0.6481 -0.6124
H -0.3416 -0.3120 0.9999
H -0.3213 1.0007 -0.2279
H -0.3304 -0.6586 -0.8008

