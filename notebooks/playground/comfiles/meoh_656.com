%nproc=4
%mem=5760MB
%chk=meoh_656.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4445 -0.0067 0.0030
C -0.0078 -0.0078 -0.0028
H 1.6616 0.9401 -0.1276
H -0.3928 -1.0183 -0.1400
H -0.2754 0.3692 0.9843
H -0.3692 0.6922 -0.7562

