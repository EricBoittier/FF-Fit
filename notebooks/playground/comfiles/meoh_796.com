%nproc=4
%mem=5760MB
%chk=meoh_796.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4185 0.0992 -0.0407
C 0.0273 -0.0052 -0.0012
H 1.8201 -0.6018 0.5141
H -0.4457 -0.4871 -0.8569
H -0.2677 -0.5873 0.8719
H -0.4890 0.9487 0.1060

