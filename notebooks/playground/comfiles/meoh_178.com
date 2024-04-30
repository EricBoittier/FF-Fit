%nproc=4
%mem=5760MB
%chk=meoh_178.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4291 0.0787 0.0527
C 0.0082 -0.0159 0.0015
H 1.7349 -0.2594 -0.8149
H -0.3177 -0.9191 -0.5145
H -0.4324 0.0326 0.9973
H -0.3069 0.8719 -0.5467

