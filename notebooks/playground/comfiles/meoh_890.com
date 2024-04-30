%nproc=4
%mem=5760MB
%chk=meoh_890.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4357 0.0422 0.0559
C -0.0080 -0.0031 -0.0090
H 1.7530 0.1913 -0.8593
H -0.4057 0.5440 -0.8638
H -0.2454 -1.0668 0.0091
H -0.3367 0.4843 0.9088

