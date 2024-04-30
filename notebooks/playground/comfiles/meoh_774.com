%nproc=4
%mem=5760MB
%chk=meoh_774.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4137 0.1091 -0.0138
C 0.0300 -0.0005 0.0222
H 1.8222 -0.7816 -0.0309
H -0.2619 -0.6293 -0.8191
H -0.3956 -0.4675 0.9104
H -0.5028 0.9380 -0.1310

