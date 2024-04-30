%nproc=4
%mem=5760MB
%chk=meoh_984.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4527 0.0390 0.0554
C -0.0115 0.0028 0.0057
H 1.5780 0.2504 -0.8933
H -0.4646 0.8354 0.5438
H -0.2147 -0.0204 -1.0649
H -0.3621 -0.9327 0.4416

