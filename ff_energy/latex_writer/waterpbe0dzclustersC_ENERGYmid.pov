#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic angle 0
  right -11.60*x up 11.19*y
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
atom(< -0.33,   0.60,  -4.62>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #0
atom(<  0.58,   0.24,  -4.88>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #1
atom(< -0.54,  -0.12,  -4.04>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #2
atom(< -2.88,   1.76,  -5.88>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #3
atom(< -1.98,   1.50,  -5.57>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #4
atom(< -2.90,   1.41,  -6.77>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #5
atom(<  4.84,  -0.38,  -2.28>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #6
atom(<  4.25,   0.28,  -2.67>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #7
atom(<  5.40,  -0.50,  -3.02>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #8
atom(< -1.20,   0.18,  -0.61>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #9
atom(< -0.52,   0.50,   0.00>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #10
atom(< -1.89,  -0.08,  -0.02>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #11
atom(<  3.97,  -1.88,  -5.97>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #12
atom(<  4.39,  -2.31,  -6.76>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #13
atom(<  4.06,  -2.67,  -5.35>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #14
atom(<  1.78,  -3.81,  -8.11>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #15
atom(<  2.26,  -3.82,  -8.93>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #16
atom(<  1.45,  -2.91,  -8.12>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #17
atom(< -0.02,   3.31,  -5.04>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #18
atom(< -0.72,   3.66,  -4.42>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #19
atom(<  0.12,   2.41,  -4.73>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #20
atom(< -0.28,  -2.28,  -6.72>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #21
atom(< -0.82,  -2.83,  -7.36>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #22
atom(<  0.20,  -1.79,  -7.44>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #23
atom(<  2.55,   4.55,  -3.59>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #24
atom(<  1.67,   4.14,  -3.76>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #25
atom(<  2.28,   5.21,  -2.99>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #26
atom(< -1.64,   0.29,  -8.02>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #27
atom(< -1.47,   1.21,  -8.22>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #28
atom(< -1.05,   0.11,  -7.31>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #29
atom(<  3.23,   0.87,  -6.19>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #30
atom(<  3.63,  -0.02,  -5.95>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #31
atom(<  3.19,   1.19,  -5.26>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #32
atom(<  1.81,   1.46,  -1.36>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #33
atom(<  1.60,   1.17,  -0.46>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #34
atom(<  1.00,   2.03,  -1.39>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #35
atom(<  0.09,  -4.41,  -4.96>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #36
atom(<  0.27,  -3.75,  -5.66>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #37
atom(< -0.20,  -5.21,  -5.48>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #38
atom(< -3.23,  -2.33,  -5.99>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #39
atom(< -3.49,  -1.61,  -5.36>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #40
atom(< -2.26,  -2.11,  -6.06>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #41
atom(< -3.68,  -0.90,  -3.60>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #42
atom(< -3.31,  -0.13,  -3.13>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #43
atom(< -2.97,  -1.52,  -3.37>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #44
atom(< -5.26,   1.56,  -4.68>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #45
atom(< -4.92,   2.27,  -4.07>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #46
atom(< -4.37,   1.31,  -5.04>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #47
atom(<  2.44,  -1.65,  -2.00>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #48
atom(<  3.27,  -1.25,  -1.80>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #49
atom(<  2.16,  -1.20,  -2.85>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #50
atom(<  3.18,   1.67,  -3.56>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #51
atom(<  3.48,   2.61,  -3.62>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #52
atom(<  2.67,   1.67,  -2.72>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #53
atom(<  2.43,   3.01,  -7.58>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #54
atom(<  2.79,   2.25,  -7.03>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #55
atom(<  2.50,   2.47,  -8.40>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #56
atom(<  1.05,  -1.08,  -8.65>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #57
atom(<  1.60,  -0.25,  -8.63>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #58
atom(<  0.79,  -0.82,  -9.58>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #59
cylinder {< -0.33,   0.60,  -4.62>, < -0.44,   0.24,  -4.33>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.54,  -0.12,  -4.04>, < -0.44,   0.24,  -4.33>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.33,   0.60,  -4.62>, <  0.12,   0.42,  -4.75>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.58,   0.24,  -4.88>, <  0.12,   0.42,  -4.75>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.88,   1.76,  -5.88>, < -2.89,   1.59,  -6.32>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.90,   1.41,  -6.77>, < -2.89,   1.59,  -6.32>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.88,   1.76,  -5.88>, < -2.43,   1.63,  -5.72>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.98,   1.50,  -5.57>, < -2.43,   1.63,  -5.72>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.84,  -0.38,  -2.28>, <  4.54,  -0.05,  -2.47>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.25,   0.28,  -2.67>, <  4.54,  -0.05,  -2.47>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.84,  -0.38,  -2.28>, <  5.12,  -0.44,  -2.65>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  5.40,  -0.50,  -3.02>, <  5.12,  -0.44,  -2.65>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.20,   0.18,  -0.61>, < -1.54,   0.05,  -0.32>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.89,  -0.08,  -0.02>, < -1.54,   0.05,  -0.32>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.20,   0.18,  -0.61>, < -0.86,   0.34,  -0.31>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.52,   0.50,   0.00>, < -0.86,   0.34,  -0.31>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.97,  -1.88,  -5.97>, <  4.02,  -2.28,  -5.66>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.06,  -2.67,  -5.35>, <  4.02,  -2.28,  -5.66>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.97,  -1.88,  -5.97>, <  4.18,  -2.10,  -6.37>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.39,  -2.31,  -6.76>, <  4.18,  -2.10,  -6.37>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.78,  -3.81,  -8.11>, <  2.02,  -3.82,  -8.52>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.26,  -3.82,  -8.93>, <  2.02,  -3.82,  -8.52>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.78,  -3.81,  -8.11>, <  1.62,  -3.36,  -8.12>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.45,  -2.91,  -8.12>, <  1.62,  -3.36,  -8.12>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.02,   3.31,  -5.04>, < -0.37,   3.49,  -4.73>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.72,   3.66,  -4.42>, < -0.37,   3.49,  -4.73>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.02,   3.31,  -5.04>, <  0.05,   2.86,  -4.88>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.12,   2.41,  -4.73>, <  0.05,   2.86,  -4.88>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.28,  -2.28,  -6.72>, < -0.04,  -2.03,  -7.08>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.20,  -1.79,  -7.44>, < -0.04,  -2.03,  -7.08>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.28,  -2.28,  -6.72>, < -0.55,  -2.56,  -7.04>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.82,  -2.83,  -7.36>, < -0.55,  -2.56,  -7.04>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.20,  -1.79,  -7.44>, <  0.63,  -1.43,  -8.05>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.05,  -1.08,  -8.65>, <  0.63,  -1.43,  -8.05>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.55,   4.55,  -3.59>, <  2.11,   4.34,  -3.67>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.67,   4.14,  -3.76>, <  2.11,   4.34,  -3.67>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.55,   4.55,  -3.59>, <  2.41,   4.88,  -3.29>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.28,   5.21,  -2.99>, <  2.41,   4.88,  -3.29>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.64,   0.29,  -8.02>, < -1.35,   0.20,  -7.66>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.05,   0.11,  -7.31>, < -1.35,   0.20,  -7.66>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.64,   0.29,  -8.02>, < -1.56,   0.75,  -8.12>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.47,   1.21,  -8.22>, < -1.56,   0.75,  -8.12>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.23,   0.87,  -6.19>, <  3.21,   1.03,  -5.73>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.19,   1.19,  -5.26>, <  3.21,   1.03,  -5.73>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.23,   0.87,  -6.19>, <  3.43,   0.42,  -6.07>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.63,  -0.02,  -5.95>, <  3.43,   0.42,  -6.07>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.81,   1.46,  -1.36>, <  1.71,   1.31,  -0.91>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.60,   1.17,  -0.46>, <  1.71,   1.31,  -0.91>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.81,   1.46,  -1.36>, <  1.40,   1.75,  -1.38>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.00,   2.03,  -1.39>, <  1.40,   1.75,  -1.38>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.81,   1.46,  -1.36>, <  2.24,   1.57,  -2.04>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.67,   1.67,  -2.72>, <  2.24,   1.57,  -2.04>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.09,  -4.41,  -4.96>, <  0.18,  -4.08,  -5.31>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.27,  -3.75,  -5.66>, <  0.18,  -4.08,  -5.31>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.09,  -4.41,  -4.96>, < -0.06,  -4.81,  -5.22>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.20,  -5.21,  -5.48>, < -0.06,  -4.81,  -5.22>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.23,  -2.33,  -5.99>, < -3.36,  -1.97,  -5.68>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.49,  -1.61,  -5.36>, < -3.36,  -1.97,  -5.68>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.23,  -2.33,  -5.99>, < -2.75,  -2.22,  -6.03>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.26,  -2.11,  -6.06>, < -2.75,  -2.22,  -6.03>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.68,  -0.90,  -3.60>, < -3.33,  -1.21,  -3.48>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.97,  -1.52,  -3.37>, < -3.33,  -1.21,  -3.48>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.68,  -0.90,  -3.60>, < -3.50,  -0.52,  -3.36>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.31,  -0.13,  -3.13>, < -3.50,  -0.52,  -3.36>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -5.26,   1.56,  -4.68>, < -5.09,   1.91,  -4.38>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -4.92,   2.27,  -4.07>, < -5.09,   1.91,  -4.38>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -5.26,   1.56,  -4.68>, < -4.81,   1.43,  -4.86>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -4.37,   1.31,  -5.04>, < -4.81,   1.43,  -4.86>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.44,  -1.65,  -2.00>, <  2.86,  -1.45,  -1.90>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.27,  -1.25,  -1.80>, <  2.86,  -1.45,  -1.90>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.44,  -1.65,  -2.00>, <  2.30,  -1.42,  -2.42>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.16,  -1.20,  -2.85>, <  2.30,  -1.42,  -2.42>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.18,   1.67,  -3.56>, <  3.33,   2.14,  -3.59>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.48,   2.61,  -3.62>, <  3.33,   2.14,  -3.59>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.18,   1.67,  -3.56>, <  2.93,   1.67,  -3.14>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.67,   1.67,  -2.72>, <  2.93,   1.67,  -3.14>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.43,   3.01,  -7.58>, <  2.61,   2.63,  -7.30>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.79,   2.25,  -7.03>, <  2.61,   2.63,  -7.30>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.43,   3.01,  -7.58>, <  2.47,   2.74,  -7.99>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.50,   2.47,  -8.40>, <  2.47,   2.74,  -7.99>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.05,  -1.08,  -8.65>, <  1.32,  -0.66,  -8.64>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.60,  -0.25,  -8.63>, <  1.32,  -0.66,  -8.64>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.05,  -1.08,  -8.65>, <  0.92,  -0.95,  -9.12>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.79,  -0.82,  -9.58>, <  0.92,  -0.95,  -9.12>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
// no constraints
