%nproc=4
%mem=5760MB
%chk=meoh_581.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4212 0.0798 -0.0667
C 0.0214 -0.0075 0.0161
H 1.7621 -0.2882 0.7753
H -0.3161 -0.9811 0.3716
H -0.3881 0.7613 0.6713
H -0.4116 0.1156 -0.9766

