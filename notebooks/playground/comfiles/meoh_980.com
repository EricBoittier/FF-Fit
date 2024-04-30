%nproc=4
%mem=5760MB
%chk=meoh_980.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4576 0.0268 0.0553
C -0.0174 -0.0018 -0.0051
H 1.5184 0.4754 -0.8139
H -0.3999 0.8799 0.5092
H -0.2266 -0.0895 -1.0712
H -0.3615 -0.8847 0.5335

