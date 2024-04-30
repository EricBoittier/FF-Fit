%nproc=4
%mem=5760MB
%chk=meoh_894.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4373 0.0567 0.0532
C -0.0267 -0.0003 -0.0039
H 1.8092 0.0004 -0.8518
H -0.3525 0.5565 -0.8825
H -0.2024 -1.0745 -0.0607
H -0.2905 0.4059 0.9726

