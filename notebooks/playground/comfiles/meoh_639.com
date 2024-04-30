%nproc=4
%mem=5760MB
%chk=meoh_639.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4218 -0.0071 -0.0337
C 0.0238 0.0057 0.0196
H 1.7293 0.8775 0.2551
H -0.3137 -1.0282 -0.0521
H -0.4308 0.4353 0.9124
H -0.3764 0.5444 -0.8393

