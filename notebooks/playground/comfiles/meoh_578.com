%nproc=4
%mem=5760MB
%chk=meoh_578.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4116 0.0802 -0.0618
C 0.0340 0.0026 0.0154
H 1.8074 -0.3555 0.7218
H -0.3019 -0.9699 0.3753
H -0.4421 0.7492 0.6510
H -0.4161 0.0567 -0.9758

