%nproc=4
%mem=5760MB
%chk=meoh_416.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4363 0.0825 0.0526
C -0.0115 -0.0213 -0.0049
H 1.7230 -0.3150 -0.7961
H -0.2972 0.8638 0.5635
H -0.3001 0.1284 -1.0453
H -0.3273 -0.9470 0.4763

