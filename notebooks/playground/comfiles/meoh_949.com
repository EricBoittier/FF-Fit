%nproc=4
%mem=5760MB
%chk=meoh_949.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4123 0.0313 -0.0536
C 0.0364 -0.0047 0.0069
H 1.8210 0.4256 0.7451
H -0.4445 0.9656 -0.1169
H -0.4260 -0.5735 -0.7998
H -0.3424 -0.4728 0.9155

