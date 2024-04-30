%nproc=4
%mem=5760MB
%chk=meoh_908.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4244 0.1114 0.0274
C 0.0339 -0.0254 0.0151
H 1.6559 -0.6297 -0.5707
H -0.3712 0.7119 -0.6780
H -0.3278 -0.9997 -0.3135
H -0.5122 0.2373 0.9211

