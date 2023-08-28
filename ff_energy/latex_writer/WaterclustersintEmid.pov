#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic angle 0
  right -9.68*x up 10.00*y
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
atom(< -0.10,   0.55,  -5.99>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #0
atom(< -0.82,  -0.08,  -5.66>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #1
atom(<  0.41,  -0.08,  -6.53>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #2
atom(< -1.75,  -1.79,  -8.51>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #3
atom(< -1.39,  -2.41,  -9.18>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #4
atom(< -1.02,  -1.73,  -7.86>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #5
atom(<  1.02,  -0.86,  -2.56>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #6
atom(<  0.35,  -0.19,  -2.33>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #7
atom(<  0.93,  -1.33,  -1.69>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #8
atom(<  4.34,   0.32,  -7.59>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #9
atom(<  3.84,   0.73,  -6.83>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #10
atom(<  3.59,   0.31,  -8.23>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #11
atom(< -1.63,   0.42, -10.17>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #12
atom(< -1.57,   1.31,  -9.73>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #13
atom(< -1.77,  -0.26,  -9.43>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #14
atom(<  2.73,   1.36,  -5.56>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #15
atom(<  1.86,   0.90,  -5.62>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #16
atom(<  2.39,   2.17,  -5.10>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #17
atom(<  4.19,   0.97,  -3.29>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #18
atom(<  3.33,   1.25,  -2.97>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #19
atom(<  4.16,   1.04,  -4.25>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #20
atom(< -2.03,  -0.98,  -4.86>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #21
atom(< -1.98,  -1.71,  -4.23>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #22
atom(< -2.96,  -0.86,  -5.02>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #23
atom(<  0.71,   3.27, -10.52>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #24
atom(<  0.36,   3.02, -11.38>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #25
atom(<  0.06,   2.88,  -9.88>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #26
atom(<  1.32,   3.94,  -6.15>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #27
atom(<  1.33,   4.26,  -7.07>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #28
atom(<  2.05,   4.47,  -5.78>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #29
atom(< -4.34,  -1.95,  -7.72>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #30
atom(< -3.46,  -2.00,  -8.19>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #31
atom(< -4.27,  -1.01,  -7.50>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #32
atom(<  0.39,  -3.29,  -5.19>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #33
atom(< -0.35,  -3.13,  -4.53>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #34
atom(<  0.02,  -4.17,  -5.52>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #35
atom(< -3.04,   3.74,  -7.04>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #36
atom(< -3.92,   3.53,  -6.70>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #37
atom(< -3.34,   4.64,  -7.37>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #38
atom(<  0.90,   3.75,  -1.74>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #39
atom(<  0.05,   3.89,  -2.19>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #40
atom(<  1.54,   3.61,  -2.48>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #41
atom(<  2.94,   3.59,  -3.60>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #42
atom(<  3.67,   3.51,  -2.99>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #43
atom(<  3.46,   3.85,  -4.40>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #44
atom(< -0.44,   3.02,  -4.38>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #45
atom(< -0.10,   3.46,  -5.14>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #46
atom(< -0.36,   2.12,  -4.76>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #47
atom(<  2.17,   0.74,  -9.12>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #48
atom(<  1.45,   0.18,  -9.24>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #49
atom(<  1.83,   1.58,  -9.58>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #50
atom(< -0.42,   0.47,  -0.57>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #51
atom(< -0.13,   1.19,   0.00>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #52
atom(< -1.33,   0.83,  -0.83>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #53
atom(< -1.44,  -3.12,  -3.23>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #54
atom(< -1.14,  -3.00,  -2.31>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #55
atom(< -1.99,  -3.96,  -3.10>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #56
atom(< -3.86,  -4.50,  -6.97>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #57
atom(< -4.29,  -4.55,  -6.03>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #58
atom(< -3.85,  -3.53,  -7.17>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #59
cylinder {< -0.10,   0.55,  -5.99>, < -0.46,   0.24,  -5.82>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.82,  -0.08,  -5.66>, < -0.46,   0.24,  -5.82>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.10,   0.55,  -5.99>, <  0.16,   0.24,  -6.26>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.41,  -0.08,  -6.53>, <  0.16,   0.24,  -6.26>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.75,  -1.79,  -8.51>, < -1.57,  -2.10,  -8.85>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.39,  -2.41,  -9.18>, < -1.57,  -2.10,  -8.85>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.75,  -1.79,  -8.51>, < -1.39,  -1.76,  -8.19>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.02,  -1.73,  -7.86>, < -1.39,  -1.76,  -8.19>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.02,  -0.86,  -2.56>, <  0.68,  -0.52,  -2.44>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.35,  -0.19,  -2.33>, <  0.68,  -0.52,  -2.44>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.02,  -0.86,  -2.56>, <  0.98,  -1.09,  -2.12>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.93,  -1.33,  -1.69>, <  0.98,  -1.09,  -2.12>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.34,   0.32,  -7.59>, <  3.97,   0.31,  -7.91>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.59,   0.31,  -8.23>, <  3.97,   0.31,  -7.91>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.34,   0.32,  -7.59>, <  4.09,   0.53,  -7.21>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.84,   0.73,  -6.83>, <  4.09,   0.53,  -7.21>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.63,   0.42, -10.17>, < -1.70,   0.08,  -9.80>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.77,  -0.26,  -9.43>, < -1.70,   0.08,  -9.80>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.63,   0.42, -10.17>, < -1.60,   0.87,  -9.95>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.57,   1.31,  -9.73>, < -1.60,   0.87,  -9.95>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.73,   1.36,  -5.56>, <  2.29,   1.13,  -5.59>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.86,   0.90,  -5.62>, <  2.29,   1.13,  -5.59>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.73,   1.36,  -5.56>, <  2.56,   1.77,  -5.33>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.39,   2.17,  -5.10>, <  2.56,   1.77,  -5.33>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.19,   0.97,  -3.29>, <  4.18,   1.01,  -3.77>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.16,   1.04,  -4.25>, <  4.18,   1.01,  -3.77>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.19,   0.97,  -3.29>, <  3.76,   1.11,  -3.13>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.33,   1.25,  -2.97>, <  3.76,   1.11,  -3.13>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.03,  -0.98,  -4.86>, < -2.50,  -0.92,  -4.94>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.96,  -0.86,  -5.02>, < -2.50,  -0.92,  -4.94>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.03,  -0.98,  -4.86>, < -2.00,  -1.34,  -4.55>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.98,  -1.71,  -4.23>, < -2.00,  -1.34,  -4.55>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.71,   3.27, -10.52>, <  0.53,   3.14, -10.95>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.36,   3.02, -11.38>, <  0.53,   3.14, -10.95>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.71,   3.27, -10.52>, <  0.39,   3.08, -10.20>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.06,   2.88,  -9.88>, <  0.39,   3.08, -10.20>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.32,   3.94,  -6.15>, <  1.32,   4.10,  -6.61>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.33,   4.26,  -7.07>, <  1.32,   4.10,  -6.61>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.32,   3.94,  -6.15>, <  1.68,   4.21,  -5.96>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.05,   4.47,  -5.78>, <  1.68,   4.21,  -5.96>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.34,  -1.95,  -7.72>, < -3.90,  -1.97,  -7.96>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.46,  -2.00,  -8.19>, < -3.90,  -1.97,  -7.96>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.34,  -1.95,  -7.72>, < -4.31,  -1.48,  -7.61>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -4.27,  -1.01,  -7.50>, < -4.31,  -1.48,  -7.61>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.39,  -3.29,  -5.19>, <  0.21,  -3.73,  -5.35>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.02,  -4.17,  -5.52>, <  0.21,  -3.73,  -5.35>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.39,  -3.29,  -5.19>, <  0.02,  -3.21,  -4.86>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.35,  -3.13,  -4.53>, <  0.02,  -3.21,  -4.86>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.04,   3.74,  -7.04>, < -3.48,   3.64,  -6.87>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.92,   3.53,  -6.70>, < -3.48,   3.64,  -6.87>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.04,   3.74,  -7.04>, < -3.19,   4.19,  -7.20>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.34,   4.64,  -7.37>, < -3.19,   4.19,  -7.20>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.90,   3.75,  -1.74>, <  0.48,   3.82,  -1.97>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.05,   3.89,  -2.19>, <  0.48,   3.82,  -1.97>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.90,   3.75,  -1.74>, <  1.22,   3.68,  -2.11>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.54,   3.61,  -2.48>, <  1.22,   3.68,  -2.11>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.94,   3.59,  -3.60>, <  3.31,   3.55,  -3.29>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.67,   3.51,  -2.99>, <  3.31,   3.55,  -3.29>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.94,   3.59,  -3.60>, <  3.20,   3.72,  -4.00>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.46,   3.85,  -4.40>, <  3.20,   3.72,  -4.00>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.44,   3.02,  -4.38>, < -0.27,   3.24,  -4.76>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.10,   3.46,  -5.14>, < -0.27,   3.24,  -4.76>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.44,   3.02,  -4.38>, < -0.40,   2.57,  -4.57>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.36,   2.12,  -4.76>, < -0.40,   2.57,  -4.57>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.17,   0.74,  -9.12>, <  1.81,   0.46,  -9.18>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.45,   0.18,  -9.24>, <  1.81,   0.46,  -9.18>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.17,   0.74,  -9.12>, <  2.00,   1.16,  -9.35>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.83,   1.58,  -9.58>, <  2.00,   1.16,  -9.35>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.42,   0.47,  -0.57>, < -0.87,   0.65,  -0.70>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.33,   0.83,  -0.83>, < -0.87,   0.65,  -0.70>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.42,   0.47,  -0.57>, < -0.27,   0.83,  -0.28>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.13,   1.19,   0.00>, < -0.27,   0.83,  -0.28>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.44,  -3.12,  -3.23>, < -1.72,  -3.54,  -3.17>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.99,  -3.96,  -3.10>, < -1.72,  -3.54,  -3.17>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.44,  -3.12,  -3.23>, < -1.29,  -3.06,  -2.77>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.14,  -3.00,  -2.31>, < -1.29,  -3.06,  -2.77>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.86,  -4.50,  -6.97>, < -3.86,  -4.01,  -7.07>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.85,  -3.53,  -7.17>, < -3.86,  -4.01,  -7.07>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.86,  -4.50,  -6.97>, < -4.08,  -4.52,  -6.50>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -4.29,  -4.55,  -6.03>, < -4.08,  -4.52,  -6.50>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
// no constraints
