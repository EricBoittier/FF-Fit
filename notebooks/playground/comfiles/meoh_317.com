%nproc=4
%mem=5760MB
%chk=meoh_317.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4125 0.0391 0.0449
C 0.0309 -0.0131 0.0156
H 1.8028 0.2903 -0.8183
H -0.2839 0.3603 -0.9589
H -0.4373 -0.9933 0.1065
H -0.4109 0.6633 0.7473

