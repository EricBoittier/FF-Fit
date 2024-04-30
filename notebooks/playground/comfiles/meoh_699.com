%nproc=4
%mem=5760MB
%chk=meoh_699.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4313 0.0298 0.0444
C -0.0073 -0.0019 0.0083
H 1.7586 0.4025 -0.8009
H -0.2637 -0.9548 -0.4547
H -0.3335 0.0599 1.0465
H -0.3341 0.8274 -0.6190

