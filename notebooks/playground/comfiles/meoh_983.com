%nproc=4
%mem=5760MB
%chk=meoh_983.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4569 0.0356 0.0563
C -0.0163 0.0025 0.0021
H 1.5452 0.3083 -0.8809
H -0.4479 0.8459 0.5412
H -0.2077 -0.0421 -1.0700
H -0.3614 -0.9215 0.4658

