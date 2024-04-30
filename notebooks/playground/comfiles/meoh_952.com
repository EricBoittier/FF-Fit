%nproc=4
%mem=5760MB
%chk=meoh_952.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4304 0.0137 -0.0544
C 0.0234 0.0034 0.0087
H 1.6881 0.6003 0.6872
H -0.4622 0.9789 -0.0173
H -0.3967 -0.5490 -0.8318
H -0.3536 -0.5038 0.8968

