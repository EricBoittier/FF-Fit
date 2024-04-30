%nproc=4
%mem=5760MB
%chk=meoh_496.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4263 0.0173 0.0330
C -0.0042 -0.0100 0.0104
H 1.8104 0.6141 -0.6429
H -0.3225 -0.3345 1.0011
H -0.3186 1.0084 -0.2182
H -0.3003 -0.6594 -0.8133

