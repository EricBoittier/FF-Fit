%nproc=4
%mem=5760MB
%chk=meoh_745.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4289 0.0937 0.0319
C 0.0013 -0.0065 0.0167
H 1.7741 -0.5196 -0.6501
H -0.2427 -0.7717 -0.7203
H -0.3716 -0.2650 1.0078
H -0.3979 0.9318 -0.3681

