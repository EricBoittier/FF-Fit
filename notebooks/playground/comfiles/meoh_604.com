%nproc=4
%mem=5760MB
%chk=meoh_604.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4258 0.0313 -0.0700
C 0.0111 0.0092 0.0185
H 1.7555 0.3114 0.8094
H -0.2591 -1.0334 0.1859
H -0.3901 0.6272 0.8217
H -0.4114 0.2732 -0.9510

