#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic angle 0
  right -11.04*x up 10.50*y
  direction 50.00*z
  location <0,0,50.00> look_at <0,0,0>}


light_source {<  2.00,   3.00,  40.00> color White
  area_light <0.70, 0, 0>, <0, 0.70, 0>, 3, 3
  adaptive 1 jitter}
// no fog
#declare simple = finish {phong 0.7}
#declare pale = finish {ambient 0.5 diffuse 0.85 roughness 0.001 specular 0.200 }
#declare intermediate = finish {ambient 0.3 diffuse 0.6 specular 0.1 roughness 0.04}
#declare vmd = finish {ambient 0.0 diffuse 0.65 phong 0.1 phong_size 40.0 specular 0.5 }
#declare jmol = finish {ambient 0.2 diffuse 0.6 specular 1 roughness 0.001 metallic}
#declare ase2 = finish {ambient 0.05 brilliance 3 diffuse 0.6 metallic specular 0.7 roughness 0.04 reflection 0.15}
#declare ase3 = finish {ambient 0.15 brilliance 2 diffuse 0.6 metallic specular 1.0 roughness 0.001 reflection 0.0}
#declare glass = finish {ambient 0.05 diffuse 0.3 specular 1.0 roughness 0.001}
#declare glass2 = finish {ambient 0.01 diffuse 0.3 specular 1.0 reflection 0.25 roughness 0.001}
#declare Rcell = 0.050;
#declare Rbond = 0.100;

#macro atom(LOC, R, COL, TRANS, FIN)
  sphere{LOC, R texture{pigment{color COL transmit TRANS} finish{FIN}}}
#end
#macro constrain(LOC, R, COL, TRANS FIN)
union{torus{R, Rcell rotate 45*z texture{pigment{color COL transmit TRANS} finish{FIN}}}
     torus{R, Rcell rotate -45*z texture{pigment{color COL transmit TRANS} finish{FIN}}}
     translate LOC}
#end

// no cell vertices
atom(<  0.33,   0.48,  -5.24>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #0
atom(< -0.29,   0.31,  -5.91>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #1
atom(< -0.10,  -0.11,  -4.59>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #2
atom(< -1.54,  -1.80,  -0.64>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #3
atom(< -2.20,  -2.19,   0.00>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #4
atom(< -1.44,  -0.87,  -0.23>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #5
atom(< -4.34,  -2.38,  -6.98>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #6
atom(< -3.68,  -1.69,  -7.05>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #7
atom(< -3.91,  -2.89,  -6.27>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #8
atom(<  4.92,   1.61,  -7.83>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #9
atom(<  4.92,   1.81,  -6.87>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #10
atom(<  4.02,   2.02,  -7.98>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #11
atom(<  4.99,   0.08,  -1.83>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #12
atom(<  4.07,   0.39,  -1.75>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #13
atom(<  4.88,  -0.26,  -2.82>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #14
atom(<  4.12,   1.87,  -5.31>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #15
atom(<  4.27,   0.92,  -5.00>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #16
atom(<  4.59,   2.36,  -4.51>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #17
atom(< -1.45,   3.19,  -5.96>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #18
atom(< -1.86,   2.65,  -6.60>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #19
atom(< -0.65,   3.47,  -6.47>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #20
atom(<  1.79,  -1.02,  -2.25>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #21
atom(<  1.62,  -1.69,  -1.59>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #22
atom(<  1.10,  -1.29,  -2.87>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #23
atom(< -4.37,   1.64,  -6.25>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #24
atom(< -4.76,   2.05,  -7.04>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #25
atom(< -5.13,   1.80,  -5.61>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #26
atom(< -4.12,  -0.08,  -3.34>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #27
atom(< -4.90,   0.40,  -3.63>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #28
atom(< -4.44,  -0.41,  -2.51>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #29
atom(< -1.22,  -0.74,  -3.35>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #30
atom(< -2.08,  -0.53,  -2.93>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #31
atom(< -1.19,  -1.71,  -3.21>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #32
atom(<  3.64,  -3.25,  -3.98>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #33
atom(<  2.89,  -3.84,  -4.11>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #34
atom(<  3.86,  -3.30,  -3.04>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #35
atom(< -3.04,   4.27,  -4.01>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #36
atom(< -3.74,   4.88,  -4.29>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #37
atom(< -2.63,   4.02,  -4.88>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #38
atom(< -4.43,   0.12,  -8.66>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #39
atom(< -4.09,  -0.25,  -9.55>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #40
atom(< -3.77,   0.91,  -8.58>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #41
atom(<  2.47,   4.38,  -2.24>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #42
atom(<  2.58,   3.40,  -2.30>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #43
atom(<  2.27,   4.47,  -1.31>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #44
atom(<  1.19,  -4.74,  -3.98>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #45
atom(<  0.63,  -4.31,  -3.30>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #46
atom(<  0.75,  -4.45,  -4.81>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #47
atom(<  2.81,   3.59,  -9.09>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #48
atom(<  2.39,   3.29,  -8.27>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #49
atom(<  2.06,   3.55,  -9.74>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #50
atom(<  1.07,  -0.47,  -9.30>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #51
atom(<  0.28,  -0.55,  -8.78>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #52
atom(<  1.06,  -1.18,  -9.94>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #53
atom(<  3.62,  -3.76,  -7.68>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #54
atom(<  3.01,  -3.87,  -8.45>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #55
atom(<  3.15,  -3.23,  -7.05>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #56
atom(< -0.28,   2.41,  -3.16>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #57
atom(< -0.59,   3.17,  -3.75>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #58
atom(< -0.09,   1.78,  -3.86>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #59
cylinder {<  0.33,   0.48,  -5.24>, <  0.11,   0.19,  -4.91>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.10,  -0.11,  -4.59>, <  0.11,   0.19,  -4.91>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.33,   0.48,  -5.24>, <  0.02,   0.39,  -5.57>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.29,   0.31,  -5.91>, <  0.02,   0.39,  -5.57>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.54,  -1.80,  -0.64>, < -1.87,  -1.99,  -0.32>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.20,  -2.19,   0.00>, < -1.87,  -1.99,  -0.32>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.54,  -1.80,  -0.64>, < -1.49,  -1.33,  -0.44>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.44,  -0.87,  -0.23>, < -1.49,  -1.33,  -0.44>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.34,  -2.38,  -6.98>, < -4.13,  -2.63,  -6.63>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.91,  -2.89,  -6.27>, < -4.13,  -2.63,  -6.63>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.34,  -2.38,  -6.98>, < -4.01,  -2.03,  -7.02>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.68,  -1.69,  -7.05>, < -4.01,  -2.03,  -7.02>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.92,   1.61,  -7.83>, <  4.92,   1.71,  -7.35>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.92,   1.81,  -6.87>, <  4.92,   1.71,  -7.35>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.92,   1.61,  -7.83>, <  4.47,   1.81,  -7.90>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.02,   2.02,  -7.98>, <  4.47,   1.81,  -7.90>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.99,   0.08,  -1.83>, <  4.94,  -0.09,  -2.32>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.88,  -0.26,  -2.82>, <  4.94,  -0.09,  -2.32>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.99,   0.08,  -1.83>, <  4.53,   0.23,  -1.79>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.07,   0.39,  -1.75>, <  4.53,   0.23,  -1.79>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.12,   1.87,  -5.31>, <  4.20,   1.40,  -5.16>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.27,   0.92,  -5.00>, <  4.20,   1.40,  -5.16>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.12,   1.87,  -5.31>, <  4.35,   2.12,  -4.91>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.59,   2.36,  -4.51>, <  4.35,   2.12,  -4.91>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.45,   3.19,  -5.96>, < -1.65,   2.92,  -6.28>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.86,   2.65,  -6.60>, < -1.65,   2.92,  -6.28>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.45,   3.19,  -5.96>, < -1.05,   3.33,  -6.22>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.65,   3.47,  -6.47>, < -1.05,   3.33,  -6.22>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.79,  -1.02,  -2.25>, <  1.70,  -1.36,  -1.92>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.62,  -1.69,  -1.59>, <  1.70,  -1.36,  -1.92>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.79,  -1.02,  -2.25>, <  1.44,  -1.16,  -2.56>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.10,  -1.29,  -2.87>, <  1.44,  -1.16,  -2.56>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.37,   1.64,  -6.25>, < -4.75,   1.72,  -5.93>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -5.13,   1.80,  -5.61>, < -4.75,   1.72,  -5.93>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.37,   1.64,  -6.25>, < -4.56,   1.85,  -6.64>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -4.76,   2.05,  -7.04>, < -4.56,   1.85,  -6.64>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.12,  -0.08,  -3.34>, < -4.28,  -0.24,  -2.93>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -4.44,  -0.41,  -2.51>, < -4.28,  -0.24,  -2.93>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.12,  -0.08,  -3.34>, < -4.51,   0.16,  -3.49>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -4.90,   0.40,  -3.63>, < -4.51,   0.16,  -3.49>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.22,  -0.74,  -3.35>, < -1.65,  -0.63,  -3.14>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.08,  -0.53,  -2.93>, < -1.65,  -0.63,  -3.14>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.22,  -0.74,  -3.35>, < -1.20,  -1.23,  -3.28>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.19,  -1.71,  -3.21>, < -1.20,  -1.23,  -3.28>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.64,  -3.25,  -3.98>, <  3.26,  -3.55,  -4.05>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.89,  -3.84,  -4.11>, <  3.26,  -3.55,  -4.05>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.64,  -3.25,  -3.98>, <  3.75,  -3.27,  -3.51>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.86,  -3.30,  -3.04>, <  3.75,  -3.27,  -3.51>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.04,   4.27,  -4.01>, < -3.39,   4.57,  -4.15>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.74,   4.88,  -4.29>, < -3.39,   4.57,  -4.15>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.04,   4.27,  -4.01>, < -2.83,   4.14,  -4.44>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.63,   4.02,  -4.88>, < -2.83,   4.14,  -4.44>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.43,   0.12,  -8.66>, < -4.10,   0.51,  -8.62>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.77,   0.91,  -8.58>, < -4.10,   0.51,  -8.62>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.43,   0.12,  -8.66>, < -4.26,  -0.06,  -9.11>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -4.09,  -0.25,  -9.55>, < -4.26,  -0.06,  -9.11>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.47,   4.38,  -2.24>, <  2.52,   3.89,  -2.27>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.58,   3.40,  -2.30>, <  2.52,   3.89,  -2.27>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.47,   4.38,  -2.24>, <  2.37,   4.42,  -1.77>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.27,   4.47,  -1.31>, <  2.37,   4.42,  -1.77>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.19,  -4.74,  -3.98>, <  0.91,  -4.52,  -3.64>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.63,  -4.31,  -3.30>, <  0.91,  -4.52,  -3.64>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.19,  -4.74,  -3.98>, <  0.97,  -4.59,  -4.39>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.75,  -4.45,  -4.81>, <  0.97,  -4.59,  -4.39>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.81,   3.59,  -9.09>, <  2.44,   3.57,  -9.42>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.06,   3.55,  -9.74>, <  2.44,   3.57,  -9.42>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.81,   3.59,  -9.09>, <  2.60,   3.44,  -8.68>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.39,   3.29,  -8.27>, <  2.60,   3.44,  -8.68>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.07,  -0.47,  -9.30>, <  0.68,  -0.51,  -9.04>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.28,  -0.55,  -8.78>, <  0.68,  -0.51,  -9.04>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.07,  -0.47,  -9.30>, <  1.07,  -0.83,  -9.62>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.06,  -1.18,  -9.94>, <  1.07,  -0.83,  -9.62>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.62,  -3.76,  -7.68>, <  3.38,  -3.49,  -7.37>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.15,  -3.23,  -7.05>, <  3.38,  -3.49,  -7.37>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.62,  -3.76,  -7.68>, <  3.31,  -3.81,  -8.06>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.01,  -3.87,  -8.45>, <  3.31,  -3.81,  -8.06>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.28,   2.41,  -3.16>, < -0.19,   2.09,  -3.51>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.09,   1.78,  -3.86>, < -0.19,   2.09,  -3.51>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.28,   2.41,  -3.16>, < -0.44,   2.79,  -3.46>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.59,   3.17,  -3.75>, < -0.44,   2.79,  -3.46>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
// no constraints
