%nproc=4
%mem=5760MB
%chk=meoh_962.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4466 -0.0050 -0.0239
C -0.0131 -0.0039 0.0037
H 1.6228 0.9284 0.2172
H -0.3185 1.0226 0.2066
H -0.3517 -0.3993 -0.9539
H -0.3002 -0.6406 0.8405

