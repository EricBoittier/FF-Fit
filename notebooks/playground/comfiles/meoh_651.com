%nproc=4
%mem=5760MB
%chk=meoh_651.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4410 -0.0044 -0.0066
C -0.0101 -0.0083 0.0053
H 1.7070 0.9387 -0.0190
H -0.3890 -1.0200 -0.1403
H -0.2711 0.3965 0.9831
H -0.3398 0.6396 -0.8068

