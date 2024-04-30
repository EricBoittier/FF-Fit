%nproc=4
%mem=5760MB
%chk=meoh_11.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4309 -0.0103 -0.0003
C 0.0090 0.0020 0.0001
H 1.7226 0.9253 0.0002
H -0.3666 -1.0213 -0.0000
H -0.3595 0.5142 0.8888
H -0.3596 0.5140 -0.8888

