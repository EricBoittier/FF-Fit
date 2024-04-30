%nproc=4
%mem=5760MB
%chk=meoh_484.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4423 -0.0110 0.0087
C 0.0118 -0.0032 -0.0001
H 1.5719 0.9342 -0.2153
H -0.4055 -0.1362 0.9981
H -0.3198 0.9843 -0.3211
H -0.4209 -0.7851 -0.6242

