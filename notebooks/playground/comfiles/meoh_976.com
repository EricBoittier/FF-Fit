%nproc=4
%mem=5760MB
%chk=meoh_976.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4338 0.0199 0.0451
C 0.0092 -0.0143 -0.0051
H 1.6425 0.6693 -0.6587
H -0.3556 0.9249 0.4109
H -0.3262 -0.1206 -1.0367
H -0.3698 -0.8345 0.6047

