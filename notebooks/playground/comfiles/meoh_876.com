%nproc=4
%mem=5760MB
%chk=meoh_876.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4351 0.0116 0.0267
C -0.0030 -0.0083 0.0177
H 1.6812 0.7279 -0.5953
H -0.2078 0.3319 -0.9974
H -0.3890 -1.0208 0.1354
H -0.3684 0.6598 0.7976

