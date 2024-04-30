%nproc=4
%mem=5760MB
%chk=meoh_871.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4356 0.0043 0.0139
C -0.0190 -0.0071 0.0177
H 1.7307 0.8350 -0.4145
H -0.1916 0.2810 -1.0194
H -0.3218 -1.0369 0.2071
H -0.3192 0.7215 0.7707

