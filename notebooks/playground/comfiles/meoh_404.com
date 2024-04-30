%nproc=4
%mem=5760MB
%chk=meoh_404.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4190 0.0266 0.0471
C 0.0383 0.0101 0.0087
H 1.7147 0.3992 -0.8098
H -0.5063 0.8886 0.3549
H -0.3316 -0.2270 -0.9888
H -0.3971 -0.8192 0.5662

