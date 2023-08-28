#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic angle 0
  right -10.69*x up 11.04*y
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
atom(<  0.27,  -0.33,  -4.40>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #0
atom(<  0.94,   0.29,  -4.57>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #1
atom(< -0.38,   0.10,  -4.98>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #2
atom(< -4.33,   1.54,  -6.67>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #3
atom(< -4.97,   2.20,  -7.06>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #4
atom(< -4.74,   1.44,  -5.75>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #5
atom(<  2.01,   4.34,  -7.25>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #6
atom(<  2.09,   3.68,  -6.57>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #7
atom(<  1.30,   3.91,  -7.77>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #8
atom(<  2.86,  -4.92,  -3.27>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #9
atom(<  1.90,  -4.92,  -3.07>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #10
atom(<  3.01,  -4.02,  -2.85>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #11
atom(< -3.14,  -4.99,  -4.80>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #12
atom(< -3.22,  -4.07,  -4.48>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #13
atom(< -2.15,  -4.88,  -5.14>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #14
atom(<  0.34,  -4.12,  -3.01>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #15
atom(<  0.03,  -4.27,  -3.95>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #16
atom(< -0.46,  -4.59,  -2.52>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #17
atom(<  0.99,   1.45,  -1.69>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #18
atom(<  1.63,   1.86,  -2.23>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #19
atom(<  1.51,   0.65,  -1.40>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #20
atom(< -2.71,  -1.79,  -5.90>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #21
atom(< -3.38,  -1.62,  -6.57>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #22
atom(< -2.10,  -1.10,  -6.17>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #23
atom(<  1.28,   4.37,  -3.24>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #24
atom(<  2.08,   4.76,  -2.82>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #25
atom(<  0.64,   5.13,  -3.07>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #26
atom(< -1.62,   4.12,  -4.96>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #27
atom(< -1.34,   4.90,  -4.48>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #28
atom(< -2.46,   4.44,  -5.28>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #29
atom(< -1.62,   1.22,  -5.62>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #30
atom(< -2.04,   2.08,  -5.40>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #31
atom(< -1.76,   1.19,  -6.59>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #32
atom(< -0.98,  -3.64,  -8.13>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #33
atom(< -0.86,  -2.89,  -8.72>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #34
atom(< -1.93,  -3.86,  -8.17>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #35
atom(< -0.96,   3.04,  -0.61>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #36
atom(< -0.68,   3.74,   0.00>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #37
atom(< -0.09,   2.63,  -0.86>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #38
atom(<  3.70,   4.43,  -4.76>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #39
atom(<  4.59,   4.09,  -5.13>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #40
atom(<  3.61,   3.77,  -3.97>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #41
atom(< -2.73,  -2.47,  -0.49>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #42
atom(< -2.67,  -2.58,  -1.47>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #43
atom(< -3.66,  -2.27,  -0.41>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #44
atom(< -0.99,  -1.19,  -9.61>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #45
atom(< -1.67,  -0.63,  -9.19>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #46
atom(< -0.16,  -0.75,  -9.32>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #47
atom(<  4.13,  -2.81,  -1.29>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #48
atom(<  3.30,  -2.39,  -1.58>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #49
atom(<  4.77,  -2.06,  -1.33>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #50
atom(<  4.33,  -1.07,  -5.34>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #51
atom(<  3.81,  -0.28,  -5.42>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #52
atom(<  4.97,  -1.06,  -6.06>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #53
atom(<  2.72,  -3.62,  -8.64>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #54
atom(<  3.48,  -3.01,  -8.75>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #55
atom(<  2.08,  -3.15,  -8.11>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #56
atom(< -1.81,   0.28,  -2.47>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #57
atom(< -1.21,   0.59,  -1.71>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #58
atom(< -1.11,   0.09,  -3.10>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #59
cylinder {<  0.27,  -0.33,  -4.40>, < -0.06,  -0.11,  -4.69>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.38,   0.10,  -4.98>, < -0.06,  -0.11,  -4.69>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.27,  -0.33,  -4.40>, <  0.60,  -0.02,  -4.48>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.94,   0.29,  -4.57>, <  0.60,  -0.02,  -4.48>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.33,   1.54,  -6.67>, < -4.65,   1.87,  -6.87>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -4.97,   2.20,  -7.06>, < -4.65,   1.87,  -6.87>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.33,   1.54,  -6.67>, < -4.53,   1.49,  -6.21>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -4.74,   1.44,  -5.75>, < -4.53,   1.49,  -6.21>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.01,   4.34,  -7.25>, <  1.66,   4.13,  -7.51>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.30,   3.91,  -7.77>, <  1.66,   4.13,  -7.51>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.01,   4.34,  -7.25>, <  2.05,   4.01,  -6.91>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.09,   3.68,  -6.57>, <  2.05,   4.01,  -6.91>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.86,  -4.92,  -3.27>, <  2.38,  -4.92,  -3.17>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.90,  -4.92,  -3.07>, <  2.38,  -4.92,  -3.17>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.86,  -4.92,  -3.27>, <  2.94,  -4.47,  -3.06>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.01,  -4.02,  -2.85>, <  2.94,  -4.47,  -3.06>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.14,  -4.99,  -4.80>, < -2.65,  -4.94,  -4.97>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.15,  -4.88,  -5.14>, < -2.65,  -4.94,  -4.97>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.14,  -4.99,  -4.80>, < -3.18,  -4.53,  -4.64>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.22,  -4.07,  -4.48>, < -3.18,  -4.53,  -4.64>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.34,  -4.12,  -3.01>, <  0.19,  -4.20,  -3.48>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.03,  -4.27,  -3.95>, <  0.19,  -4.20,  -3.48>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.34,  -4.12,  -3.01>, < -0.06,  -4.35,  -2.76>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.46,  -4.59,  -2.52>, < -0.06,  -4.35,  -2.76>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.99,   1.45,  -1.69>, <  1.31,   1.65,  -1.96>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.63,   1.86,  -2.23>, <  1.31,   1.65,  -1.96>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.99,   1.45,  -1.69>, <  1.25,   1.05,  -1.55>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.51,   0.65,  -1.40>, <  1.25,   1.05,  -1.55>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.71,  -1.79,  -5.90>, < -3.04,  -1.70,  -6.23>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.38,  -1.62,  -6.57>, < -3.04,  -1.70,  -6.23>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.71,  -1.79,  -5.90>, < -2.41,  -1.44,  -6.04>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.10,  -1.10,  -6.17>, < -2.41,  -1.44,  -6.04>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.28,   4.37,  -3.24>, <  0.96,   4.75,  -3.16>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.64,   5.13,  -3.07>, <  0.96,   4.75,  -3.16>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.28,   4.37,  -3.24>, <  1.68,   4.56,  -3.03>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.08,   4.76,  -2.82>, <  1.68,   4.56,  -3.03>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.62,   4.12,  -4.96>, < -2.04,   4.28,  -5.12>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.46,   4.44,  -5.28>, < -2.04,   4.28,  -5.12>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.62,   4.12,  -4.96>, < -1.48,   4.51,  -4.72>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.34,   4.90,  -4.48>, < -1.48,   4.51,  -4.72>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.62,   1.22,  -5.62>, < -1.83,   1.65,  -5.51>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.04,   2.08,  -5.40>, < -1.83,   1.65,  -5.51>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.62,   1.22,  -5.62>, < -1.69,   1.20,  -6.10>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.76,   1.19,  -6.59>, < -1.69,   1.20,  -6.10>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.98,  -3.64,  -8.13>, < -0.92,  -3.26,  -8.42>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.86,  -2.89,  -8.72>, < -0.92,  -3.26,  -8.42>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.98,  -3.64,  -8.13>, < -1.45,  -3.75,  -8.15>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.93,  -3.86,  -8.17>, < -1.45,  -3.75,  -8.15>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.96,   3.04,  -0.61>, < -0.82,   3.39,  -0.31>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.68,   3.74,   0.00>, < -0.82,   3.39,  -0.31>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.96,   3.04,  -0.61>, < -0.53,   2.83,  -0.73>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.09,   2.63,  -0.86>, < -0.53,   2.83,  -0.73>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.70,   4.43,  -4.76>, <  3.65,   4.10,  -4.36>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.61,   3.77,  -3.97>, <  3.65,   4.10,  -4.36>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.70,   4.43,  -4.76>, <  4.14,   4.26,  -4.94>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.59,   4.09,  -5.13>, <  4.14,   4.26,  -4.94>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.73,  -2.47,  -0.49>, < -2.70,  -2.52,  -0.98>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.67,  -2.58,  -1.47>, < -2.70,  -2.52,  -0.98>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.73,  -2.47,  -0.49>, < -3.19,  -2.37,  -0.45>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.66,  -2.27,  -0.41>, < -3.19,  -2.37,  -0.45>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.99,  -1.19,  -9.61>, < -1.33,  -0.91,  -9.40>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.67,  -0.63,  -9.19>, < -1.33,  -0.91,  -9.40>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.99,  -1.19,  -9.61>, < -0.57,  -0.97,  -9.47>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.16,  -0.75,  -9.32>, < -0.57,  -0.97,  -9.47>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.13,  -2.81,  -1.29>, <  4.45,  -2.44,  -1.31>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.77,  -2.06,  -1.33>, <  4.45,  -2.44,  -1.31>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.13,  -2.81,  -1.29>, <  3.71,  -2.60,  -1.44>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.30,  -2.39,  -1.58>, <  3.71,  -2.60,  -1.44>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.33,  -1.07,  -5.34>, <  4.07,  -0.68,  -5.38>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.81,  -0.28,  -5.42>, <  4.07,  -0.68,  -5.38>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.33,  -1.07,  -5.34>, <  4.65,  -1.07,  -5.70>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.97,  -1.06,  -6.06>, <  4.65,  -1.07,  -5.70>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.72,  -3.62,  -8.64>, <  2.40,  -3.38,  -8.37>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.08,  -3.15,  -8.11>, <  2.40,  -3.38,  -8.37>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.72,  -3.62,  -8.64>, <  3.10,  -3.31,  -8.69>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.48,  -3.01,  -8.75>, <  3.10,  -3.31,  -8.69>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.81,   0.28,  -2.47>, < -1.46,   0.19,  -2.78>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.11,   0.09,  -3.10>, < -1.46,   0.19,  -2.78>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.81,   0.28,  -2.47>, < -1.51,   0.44,  -2.09>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.21,   0.59,  -1.71>, < -1.51,   0.44,  -2.09>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
// no constraints
