#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic angle 0
  right -11.07*x up 12.30*y
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
atom(< -0.49,  -0.84,  -4.10>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #0
atom(< -0.55,  -0.02,  -3.60>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #1
atom(< -0.52,  -0.57,  -5.04>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #2
atom(<  2.04,  -1.25,  -5.65>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #3
atom(<  1.65,  -2.08,  -5.81>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #4
atom(<  2.13,  -0.95,  -6.53>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #5
atom(< -1.52,  -3.13,  -0.71>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #6
atom(< -2.05,  -3.53,   0.00>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #7
atom(< -2.23,  -2.51,  -0.95>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #8
atom(<  3.06,   3.40,  -3.89>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #9
atom(<  4.02,   3.47,  -4.12>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #10
atom(<  2.90,   2.54,  -4.23>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #11
atom(< -3.18,  -1.75,  -3.54>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #12
atom(< -2.31,  -1.38,  -3.69>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #13
atom(< -3.27,  -1.56,  -2.57>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #14
atom(< -1.14,   5.59,  -4.11>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #15
atom(< -2.01,   5.26,  -4.47>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #16
atom(< -1.14,   4.89,  -3.40>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #17
atom(<  4.26,  -0.22,  -3.48>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #18
atom(<  4.25,  -0.04,  -4.49>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #19
atom(<  5.15,  -0.70,  -3.44>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #20
atom(< -0.34,   1.52,  -1.61>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #21
atom(< -0.07,   1.20,  -0.76>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #22
atom(<  0.39,   1.19,  -2.10>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #23
atom(< -3.57,   3.91,  -5.05>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #24
atom(< -3.68,   4.42,  -5.82>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #25
atom(< -4.46,   3.53,  -4.83>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #26
atom(< -4.92,  -3.31,  -5.13>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #27
atom(< -5.02,  -2.69,  -5.83>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #28
atom(< -4.53,  -2.70,  -4.43>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #29
atom(< -1.46,  -5.48,  -2.55>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #30
atom(< -0.56,  -5.73,  -2.38>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #31
atom(< -1.61,  -4.82,  -1.83>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #32
atom(< -4.90,  -1.41,  -7.12>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #33
atom(< -5.15,  -1.18,  -8.06>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #34
atom(< -4.45,  -0.58,  -6.96>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #35
atom(<  2.88,  -4.81,  -3.99>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #36
atom(<  2.38,  -4.21,  -3.43>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #37
atom(<  2.76,  -4.44,  -4.90>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #38
atom(< -1.61,   1.89,  -4.83>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #39
atom(< -2.25,   2.63,  -4.68>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #40
atom(< -1.27,   2.07,  -5.75>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #41
atom(<  4.25,  -0.82,  -0.65>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #42
atom(<  5.05,  -0.79,  -1.23>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #43
atom(<  3.61,  -0.33,  -1.20>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #44
atom(< -1.35,   3.97,  -2.06>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #45
atom(< -1.90,   3.87,  -1.27>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #46
atom(< -0.74,   3.23,  -2.03>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #47
atom(< -3.29,   0.63,  -7.32>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #48
atom(< -2.45,   1.15,  -7.26>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #49
atom(< -3.94,   1.35,  -7.04>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #50
atom(<  0.01,  -3.55,  -7.72>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #51
atom(< -0.55,  -3.93,  -7.02>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #52
atom(< -0.57,  -2.71,  -7.96>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #53
atom(< -3.67,   1.50,  -2.37>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #54
atom(< -3.01,   1.69,  -3.08>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #55
atom(< -3.83,   2.43,  -1.98>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #56
atom(< -1.12,   2.70,  -7.65>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #57
atom(< -0.82,   2.40,  -8.52>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #58
atom(< -0.24,   2.99,  -7.26>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #59
cylinder {< -0.49,  -0.84,  -4.10>, < -0.51,  -0.70,  -4.57>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.52,  -0.57,  -5.04>, < -0.51,  -0.70,  -4.57>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.49,  -0.84,  -4.10>, < -0.52,  -0.43,  -3.85>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.55,  -0.02,  -3.60>, < -0.52,  -0.43,  -3.85>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.04,  -1.25,  -5.65>, <  2.08,  -1.10,  -6.09>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.13,  -0.95,  -6.53>, <  2.08,  -1.10,  -6.09>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.04,  -1.25,  -5.65>, <  1.84,  -1.66,  -5.73>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.65,  -2.08,  -5.81>, <  1.84,  -1.66,  -5.73>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.52,  -3.13,  -0.71>, < -1.87,  -2.82,  -0.83>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.23,  -2.51,  -0.95>, < -1.87,  -2.82,  -0.83>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.52,  -3.13,  -0.71>, < -1.78,  -3.33,  -0.35>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.05,  -3.53,   0.00>, < -1.78,  -3.33,  -0.35>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.06,   3.40,  -3.89>, <  3.54,   3.44,  -4.01>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.02,   3.47,  -4.12>, <  3.54,   3.44,  -4.01>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.06,   3.40,  -3.89>, <  2.98,   2.97,  -4.06>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.90,   2.54,  -4.23>, <  2.98,   2.97,  -4.06>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.18,  -1.75,  -3.54>, < -3.23,  -1.66,  -3.06>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.27,  -1.56,  -2.57>, < -3.23,  -1.66,  -3.06>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.18,  -1.75,  -3.54>, < -2.75,  -1.57,  -3.62>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.31,  -1.38,  -3.69>, < -2.75,  -1.57,  -3.62>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.14,   5.59,  -4.11>, < -1.14,   5.24,  -3.76>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.14,   4.89,  -3.40>, < -1.14,   5.24,  -3.76>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.14,   5.59,  -4.11>, < -1.58,   5.43,  -4.29>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.01,   5.26,  -4.47>, < -1.58,   5.43,  -4.29>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.14,   4.89,  -3.40>, < -1.25,   4.43,  -2.73>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.35,   3.97,  -2.06>, < -1.25,   4.43,  -2.73>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.26,  -0.22,  -3.48>, <  4.70,  -0.46,  -3.46>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  5.15,  -0.70,  -3.44>, <  4.70,  -0.46,  -3.46>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.26,  -0.22,  -3.48>, <  4.25,  -0.13,  -3.99>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.25,  -0.04,  -4.49>, <  4.25,  -0.13,  -3.99>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.34,   1.52,  -1.61>, <  0.02,   1.36,  -1.86>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.39,   1.19,  -2.10>, <  0.02,   1.36,  -1.86>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.34,   1.52,  -1.61>, < -0.21,   1.36,  -1.19>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.07,   1.20,  -0.76>, < -0.21,   1.36,  -1.19>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.57,   3.91,  -5.05>, < -3.62,   4.17,  -5.44>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.68,   4.42,  -5.82>, < -3.62,   4.17,  -5.44>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.57,   3.91,  -5.05>, < -4.01,   3.72,  -4.94>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -4.46,   3.53,  -4.83>, < -4.01,   3.72,  -4.94>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.92,  -3.31,  -5.13>, < -4.97,  -3.00,  -5.48>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -5.02,  -2.69,  -5.83>, < -4.97,  -3.00,  -5.48>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.92,  -3.31,  -5.13>, < -4.72,  -3.01,  -4.78>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -4.53,  -2.70,  -4.43>, < -4.72,  -3.01,  -4.78>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.46,  -5.48,  -2.55>, < -1.01,  -5.61,  -2.47>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.56,  -5.73,  -2.38>, < -1.01,  -5.61,  -2.47>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.46,  -5.48,  -2.55>, < -1.53,  -5.15,  -2.19>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.61,  -4.82,  -1.83>, < -1.53,  -5.15,  -2.19>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.90,  -1.41,  -7.12>, < -5.02,  -1.29,  -7.59>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -5.15,  -1.18,  -8.06>, < -5.02,  -1.29,  -7.59>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.90,  -1.41,  -7.12>, < -4.67,  -0.99,  -7.04>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -4.45,  -0.58,  -6.96>, < -4.67,  -0.99,  -7.04>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.88,  -4.81,  -3.99>, <  2.82,  -4.62,  -4.45>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.76,  -4.44,  -4.90>, <  2.82,  -4.62,  -4.45>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.88,  -4.81,  -3.99>, <  2.63,  -4.51,  -3.71>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.38,  -4.21,  -3.43>, <  2.63,  -4.51,  -3.71>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.61,   1.89,  -4.83>, < -1.44,   1.98,  -5.29>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.27,   2.07,  -5.75>, < -1.44,   1.98,  -5.29>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.61,   1.89,  -4.83>, < -1.93,   2.26,  -4.76>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.25,   2.63,  -4.68>, < -1.93,   2.26,  -4.76>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.25,  -0.82,  -0.65>, <  3.93,  -0.57,  -0.92>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.61,  -0.33,  -1.20>, <  3.93,  -0.57,  -0.92>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.25,  -0.82,  -0.65>, <  4.65,  -0.81,  -0.94>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  5.05,  -0.79,  -1.23>, <  4.65,  -0.81,  -0.94>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.35,   3.97,  -2.06>, < -1.05,   3.60,  -2.04>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.74,   3.23,  -2.03>, < -1.05,   3.60,  -2.04>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.35,   3.97,  -2.06>, < -1.62,   3.92,  -1.66>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.90,   3.87,  -1.27>, < -1.62,   3.92,  -1.66>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.29,   0.63,  -7.32>, < -2.87,   0.89,  -7.29>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.45,   1.15,  -7.26>, < -2.87,   0.89,  -7.29>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.29,   0.63,  -7.32>, < -3.62,   0.99,  -7.18>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.94,   1.35,  -7.04>, < -3.62,   0.99,  -7.18>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.01,  -3.55,  -7.72>, < -0.27,  -3.74,  -7.37>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.55,  -3.93,  -7.02>, < -0.27,  -3.74,  -7.37>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.01,  -3.55,  -7.72>, < -0.28,  -3.13,  -7.84>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.57,  -2.71,  -7.96>, < -0.28,  -3.13,  -7.84>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.67,   1.50,  -2.37>, < -3.34,   1.60,  -2.72>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.01,   1.69,  -3.08>, < -3.34,   1.60,  -2.72>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.67,   1.50,  -2.37>, < -3.75,   1.96,  -2.17>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.83,   2.43,  -1.98>, < -3.75,   1.96,  -2.17>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.12,   2.70,  -7.65>, < -0.68,   2.84,  -7.45>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.24,   2.99,  -7.26>, < -0.68,   2.84,  -7.45>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.12,   2.70,  -7.65>, < -0.97,   2.55,  -8.08>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.82,   2.40,  -8.52>, < -0.97,   2.55,  -8.08>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
// no constraints
