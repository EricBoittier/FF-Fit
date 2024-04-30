%nproc=4
%mem=5760MB
%chk=meoh_617.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4250 0.0207 -0.0573
C 0.0190 -0.0139 0.0087
H 1.7514 0.5913 0.6696
H -0.3985 -1.0117 0.1435
H -0.3256 0.5925 0.8463
H -0.4137 0.4492 -0.8781

