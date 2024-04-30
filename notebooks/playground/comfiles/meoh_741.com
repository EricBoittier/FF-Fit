%nproc=4
%mem=5760MB
%chk=meoh_741.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4397 0.0910 0.0381
C -0.0118 -0.0056 0.0145
H 1.7080 -0.4681 -0.7208
H -0.2062 -0.8047 -0.7009
H -0.3679 -0.2509 1.0150
H -0.3874 0.9316 -0.3960

