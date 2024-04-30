%nproc=4
%mem=5760MB
%chk=meoh_844.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4224 0.0115 -0.0512
C 0.0133 -0.0094 0.0226
H 1.7796 0.7053 0.5418
H -0.3178 0.0005 -1.0159
H -0.3755 -0.8996 0.5169
H -0.3631 0.9076 0.4759

