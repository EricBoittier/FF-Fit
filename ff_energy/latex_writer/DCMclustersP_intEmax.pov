#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic angle 0
  right -19.38*x up 21.27*y
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
atom(< -0.19,   6.21,  -2.88>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #0
atom(< -1.44,   7.12,  -3.75>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #1
atom(<  0.50,   7.38,  -1.74>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #2
atom(<  0.58,   5.83,  -3.60>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #3
atom(< -0.66,   5.38,  -2.31>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #4
atom(<  0.78,  -1.63,  -0.97>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #5
atom(<  2.53,  -1.63,  -1.29>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #6
atom(<  0.24,   0.10,  -0.79>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #7
atom(<  0.64,  -2.16,   0.00>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #8
atom(<  0.17,  -2.02,  -1.80>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #9
atom(< -2.75,  -6.74,  -6.92>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #10
atom(< -3.16,  -5.88,  -5.45>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #11
atom(< -1.19,  -6.04,  -7.54>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #12
atom(< -3.53,  -6.49,  -7.65>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #13
atom(< -2.64,  -7.83,  -6.71>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #14
atom(< -2.45,   4.89,  -6.70>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #15
atom(< -0.66,   4.83,  -6.95>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #16
atom(< -2.88,   4.30,  -5.07>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #17
atom(< -3.03,   4.28,  -7.46>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #18
atom(< -2.74,   5.94,  -6.73>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #19
atom(<  2.29,   3.42,  -1.79>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #20
atom(<  3.67,   2.34,  -1.30>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #21
atom(<  2.82,   4.43,  -3.13>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #22
atom(<  1.92,   3.99,  -0.90>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #23
atom(<  1.49,   2.81,  -2.16>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #24
atom(<  7.72,   1.47,  -7.20>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #25
atom(<  6.21,   1.47,  -6.31>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #26
atom(<  8.82,   2.62,  -6.28>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #27
atom(<  7.52,   1.79,  -8.24>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #28
atom(<  8.05,   0.40,  -7.09>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #29
atom(<  4.29,  -8.07,  -6.95>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #30
atom(<  4.92,  -9.72,  -7.34>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #31
atom(<  3.14,  -8.40,  -5.63>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #32
atom(<  3.72,  -7.63,  -7.75>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #33
atom(<  5.12,  -7.38,  -6.63>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #34
atom(<  2.55,  -3.70,  -7.64>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #35
atom(<  4.18,  -3.34,  -6.96>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #36
atom(<  2.67,  -5.24,  -8.54>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #37
atom(<  2.16,  -2.90,  -8.26>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #38
atom(<  1.85,  -3.77,  -6.80>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #39
atom(<  3.78,   5.23,  -7.35>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #40
atom(<  3.01,   3.84,  -6.50>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #41
atom(<  3.02,   6.89,  -7.11>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #42
atom(<  3.85,   5.05,  -8.45>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #43
atom(<  4.83,   5.36,  -6.95>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #44
atom(<  0.22,   3.72, -10.30>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #45
atom(< -0.31,   3.39, -11.95>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #46
atom(<  1.13,   5.26, -10.35>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #47
atom(<  0.85,   2.95,  -9.82>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #48
atom(< -0.67,   3.89,  -9.69>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #49
atom(< -1.10,  -5.16,  -2.25>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #50
atom(< -1.92,  -3.65,  -2.57>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #51
atom(< -1.90,  -5.96,  -0.92>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #52
atom(< -0.11,  -4.95,  -1.91>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #53
atom(< -1.07,  -5.76,  -3.19>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #54
atom(< -1.14,  -3.42, -11.86>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #55
atom(<  0.47,  -3.14, -11.08>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #56
atom(< -1.14,  -5.04, -12.71>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #57
atom(< -1.27,  -2.63, -12.61>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #58
atom(< -1.98,  -3.42, -11.09>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #59
atom(< -0.47,   8.44,  -7.77>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #60
atom(<  0.40,   9.72,  -6.83>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #61
atom(< -2.07,   8.98,  -8.33>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #62
atom(< -0.62,   7.59,  -7.08>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #63
atom(<  0.15,   8.17,  -8.66>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #64
atom(<  3.80,   4.18, -12.42>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #65
atom(<  4.16,   3.33, -10.85>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #66
atom(<  5.17,   5.36, -12.62>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #67
atom(<  2.86,   4.81, -12.34>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #68
atom(<  3.74,   3.54, -13.35>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #69
atom(< -1.64,  -1.26,  -8.52>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #70
atom(< -2.71,  -0.61,  -9.76>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #71
atom(< -0.31,  -0.11,  -8.45>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #72
atom(< -1.26,  -2.24,  -8.79>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #73
atom(< -2.25,  -1.29,  -7.57>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #74
atom(<  2.57,  -0.38,  -5.28>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #75
atom(<  1.39,   0.95,  -5.26>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #76
atom(<  1.64,  -1.87,  -4.89>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #77
atom(<  3.29,  -0.24,  -4.43>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #78
atom(<  3.12,  -0.45,  -6.21>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #79
atom(<  0.75,  -7.62, -10.56>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #80
atom(<  0.04,  -8.30, -12.11>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #81
atom(<  2.20,  -8.51,  -9.94>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #82
atom(<  0.00,  -7.65,  -9.77>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #83
atom(<  0.96,  -6.57, -10.77>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #84
atom(< -2.22,   0.50,  -4.16>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #85
atom(< -1.79,  -1.10,  -4.77>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #86
atom(< -3.98,   0.73,  -4.37>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #87
atom(< -1.68,   1.34,  -4.65>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #88
atom(< -2.01,   0.41,  -3.07>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #89
atom(<  3.83,  -0.48, -11.59>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #90
atom(<  4.77,  -1.18, -10.24>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #91
atom(<  2.21,   0.10, -10.94>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #92
atom(<  3.70,  -1.30, -12.30>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #93
atom(<  4.39,   0.38, -11.99>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #94
atom(<  7.00,  -1.74,  -4.56>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #95
atom(<  8.68,  -1.16,  -4.63>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #96
atom(<  6.91,  -3.41,  -3.78>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #97
atom(<  6.54,  -1.72,  -5.56>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #98
atom(<  6.46,  -1.06,  -3.87>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #99
cylinder {< -0.19,   6.21,  -2.88>, <  0.19,   6.02,  -3.24>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.58,   5.83,  -3.60>, <  0.19,   6.02,  -3.24>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.19,   6.21,  -2.88>, < -0.42,   5.79,  -2.59>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.66,   5.38,  -2.31>, < -0.42,   5.79,  -2.59>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.19,   6.21,  -2.88>, < -0.82,   6.67,  -3.32>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.44,   7.12,  -3.75>, < -0.82,   6.67,  -3.32>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -0.19,   6.21,  -2.88>, <  0.15,   6.80,  -2.31>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.50,   7.38,  -1.74>, <  0.15,   6.80,  -2.31>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  0.78,  -1.63,  -0.97>, <  0.51,  -0.76,  -0.88>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.24,   0.10,  -0.79>, <  0.51,  -0.76,  -0.88>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  0.78,  -1.63,  -0.97>, <  0.71,  -1.89,  -0.49>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.64,  -2.16,   0.00>, <  0.71,  -1.89,  -0.49>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.78,  -1.63,  -0.97>, <  0.48,  -1.83,  -1.38>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.17,  -2.02,  -1.80>, <  0.48,  -1.83,  -1.38>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.78,  -1.63,  -0.97>, <  1.66,  -1.63,  -1.13>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.53,  -1.63,  -1.29>, <  1.66,  -1.63,  -1.13>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -2.75,  -6.74,  -6.92>, < -3.14,  -6.61,  -7.28>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.53,  -6.49,  -7.65>, < -3.14,  -6.61,  -7.28>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.75,  -6.74,  -6.92>, < -1.97,  -6.39,  -7.23>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.19,  -6.04,  -7.54>, < -1.97,  -6.39,  -7.23>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -2.75,  -6.74,  -6.92>, < -2.95,  -6.31,  -6.19>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.16,  -5.88,  -5.45>, < -2.95,  -6.31,  -6.19>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -2.75,  -6.74,  -6.92>, < -2.69,  -7.28,  -6.82>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.64,  -7.83,  -6.71>, < -2.69,  -7.28,  -6.82>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.45,   4.89,  -6.70>, < -2.74,   4.59,  -7.08>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.03,   4.28,  -7.46>, < -2.74,   4.59,  -7.08>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.45,   4.89,  -6.70>, < -1.56,   4.86,  -6.82>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.66,   4.83,  -6.95>, < -1.56,   4.86,  -6.82>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -2.45,   4.89,  -6.70>, < -2.59,   5.41,  -6.72>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.74,   5.94,  -6.73>, < -2.59,   5.41,  -6.72>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.45,   4.89,  -6.70>, < -2.66,   4.59,  -5.88>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.88,   4.30,  -5.07>, < -2.66,   4.59,  -5.88>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  2.29,   3.42,  -1.79>, <  2.55,   3.93,  -2.46>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.82,   4.43,  -3.13>, <  2.55,   3.93,  -2.46>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  2.29,   3.42,  -1.79>, <  1.89,   3.11,  -1.97>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.49,   2.81,  -2.16>, <  1.89,   3.11,  -1.97>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.29,   3.42,  -1.79>, <  2.10,   3.71,  -1.34>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.92,   3.99,  -0.90>, <  2.10,   3.71,  -1.34>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.29,   3.42,  -1.79>, <  2.98,   2.88,  -1.54>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.67,   2.34,  -1.30>, <  2.98,   2.88,  -1.54>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  7.72,   1.47,  -7.20>, <  7.62,   1.63,  -7.72>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  7.52,   1.79,  -8.24>, <  7.62,   1.63,  -7.72>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  7.72,   1.47,  -7.20>, <  7.88,   0.93,  -7.15>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  8.05,   0.40,  -7.09>, <  7.88,   0.93,  -7.15>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  7.72,   1.47,  -7.20>, <  6.96,   1.47,  -6.76>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  6.21,   1.47,  -6.31>, <  6.96,   1.47,  -6.76>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  7.72,   1.47,  -7.20>, <  8.27,   2.04,  -6.74>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  8.82,   2.62,  -6.28>, <  8.27,   2.04,  -6.74>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  4.29,  -8.07,  -6.95>, <  4.01,  -7.85,  -7.35>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.72,  -7.63,  -7.75>, <  4.01,  -7.85,  -7.35>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.29,  -8.07,  -6.95>, <  4.61,  -8.90,  -7.15>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.92,  -9.72,  -7.34>, <  4.61,  -8.90,  -7.15>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  4.29,  -8.07,  -6.95>, <  3.72,  -8.24,  -6.29>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.14,  -8.40,  -5.63>, <  3.72,  -8.24,  -6.29>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  4.29,  -8.07,  -6.95>, <  4.70,  -7.73,  -6.79>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.12,  -7.38,  -6.63>, <  4.70,  -7.73,  -6.79>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.55,  -3.70,  -7.64>, <  2.36,  -3.30,  -7.95>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.16,  -2.90,  -8.26>, <  2.36,  -3.30,  -7.95>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.55,  -3.70,  -7.64>, <  3.36,  -3.52,  -7.30>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.18,  -3.34,  -6.96>, <  3.36,  -3.52,  -7.30>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  2.55,  -3.70,  -7.64>, <  2.61,  -4.47,  -8.09>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.67,  -5.24,  -8.54>, <  2.61,  -4.47,  -8.09>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  2.55,  -3.70,  -7.64>, <  2.20,  -3.74,  -7.22>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.85,  -3.77,  -6.80>, <  2.20,  -3.74,  -7.22>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.78,   5.23,  -7.35>, <  3.81,   5.14,  -7.90>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.85,   5.05,  -8.45>, <  3.81,   5.14,  -7.90>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.78,   5.23,  -7.35>, <  3.40,   6.06,  -7.23>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.02,   6.89,  -7.11>, <  3.40,   6.06,  -7.23>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.78,   5.23,  -7.35>, <  3.39,   4.53,  -6.93>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.01,   3.84,  -6.50>, <  3.39,   4.53,  -6.93>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.78,   5.23,  -7.35>, <  4.30,   5.29,  -7.15>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.83,   5.36,  -6.95>, <  4.30,   5.29,  -7.15>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.22,   3.72, -10.30>, < -0.22,   3.81,  -9.99>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.67,   3.89,  -9.69>, < -0.22,   3.81,  -9.99>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.22,   3.72, -10.30>, <  0.53,   3.33, -10.06>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.85,   2.95,  -9.82>, <  0.53,   3.33, -10.06>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.22,   3.72, -10.30>, < -0.05,   3.56, -11.12>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.31,   3.39, -11.95>, < -0.05,   3.56, -11.12>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  0.22,   3.72, -10.30>, <  0.68,   4.49, -10.33>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.13,   5.26, -10.35>, <  0.68,   4.49, -10.33>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.10,  -5.16,  -2.25>, < -1.50,  -5.56,  -1.59>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.90,  -5.96,  -0.92>, < -1.50,  -5.56,  -1.59>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.10,  -5.16,  -2.25>, < -1.51,  -4.41,  -2.41>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.92,  -3.65,  -2.57>, < -1.51,  -4.41,  -2.41>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.10,  -5.16,  -2.25>, < -1.09,  -5.46,  -2.72>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.07,  -5.76,  -3.19>, < -1.09,  -5.46,  -2.72>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.10,  -5.16,  -2.25>, < -0.60,  -5.06,  -2.08>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.11,  -4.95,  -1.91>, < -0.60,  -5.06,  -2.08>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.14,  -3.42, -11.86>, < -1.20,  -3.02, -12.23>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.27,  -2.63, -12.61>, < -1.20,  -3.02, -12.23>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.14,  -3.42, -11.86>, < -0.34,  -3.28, -11.47>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.47,  -3.14, -11.08>, < -0.34,  -3.28, -11.47>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.14,  -3.42, -11.86>, < -1.56,  -3.42, -11.48>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.98,  -3.42, -11.09>, < -1.56,  -3.42, -11.48>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.14,  -3.42, -11.86>, < -1.14,  -4.23, -12.29>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.14,  -5.04, -12.71>, < -1.14,  -4.23, -12.29>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -0.47,   8.44,  -7.77>, < -0.55,   8.01,  -7.43>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.62,   7.59,  -7.08>, < -0.55,   8.01,  -7.43>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.47,   8.44,  -7.77>, < -1.27,   8.71,  -8.05>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.07,   8.98,  -8.33>, < -1.27,   8.71,  -8.05>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -0.47,   8.44,  -7.77>, < -0.16,   8.30,  -8.21>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.15,   8.17,  -8.66>, < -0.16,   8.30,  -8.21>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.47,   8.44,  -7.77>, < -0.04,   9.08,  -7.30>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.40,   9.72,  -6.83>, < -0.04,   9.08,  -7.30>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.80,   4.18, -12.42>, <  3.98,   3.75, -11.64>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.16,   3.33, -10.85>, <  3.98,   3.75, -11.64>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.80,   4.18, -12.42>, <  4.48,   4.77, -12.52>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.17,   5.36, -12.62>, <  4.48,   4.77, -12.52>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.80,   4.18, -12.42>, <  3.33,   4.50, -12.38>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.86,   4.81, -12.34>, <  3.33,   4.50, -12.38>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.80,   4.18, -12.42>, <  3.77,   3.86, -12.89>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.74,   3.54, -13.35>, <  3.77,   3.86, -12.89>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.64,  -1.26,  -8.52>, < -0.97,  -0.68,  -8.48>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.31,  -0.11,  -8.45>, < -0.97,  -0.68,  -8.48>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.64,  -1.26,  -8.52>, < -1.95,  -1.28,  -8.04>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.25,  -1.29,  -7.57>, < -1.95,  -1.28,  -8.04>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.64,  -1.26,  -8.52>, < -2.18,  -0.94,  -9.14>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.71,  -0.61,  -9.76>, < -2.18,  -0.94,  -9.14>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.64,  -1.26,  -8.52>, < -1.45,  -1.75,  -8.65>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.26,  -2.24,  -8.79>, < -1.45,  -1.75,  -8.65>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.57,  -0.38,  -5.28>, <  1.98,   0.29,  -5.27>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.39,   0.95,  -5.26>, <  1.98,   0.29,  -5.27>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  2.57,  -0.38,  -5.28>, <  2.93,  -0.31,  -4.85>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.29,  -0.24,  -4.43>, <  2.93,  -0.31,  -4.85>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.57,  -0.38,  -5.28>, <  2.10,  -1.12,  -5.08>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.64,  -1.87,  -4.89>, <  2.10,  -1.12,  -5.08>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  2.57,  -0.38,  -5.28>, <  2.84,  -0.41,  -5.74>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.12,  -0.45,  -6.21>, <  2.84,  -0.41,  -5.74>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.75,  -7.62, -10.56>, <  0.85,  -7.10, -10.66>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.96,  -6.57, -10.77>, <  0.85,  -7.10, -10.66>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.75,  -7.62, -10.56>, <  0.38,  -7.64, -10.16>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.00,  -7.65,  -9.77>, <  0.38,  -7.64, -10.16>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.75,  -7.62, -10.56>, <  0.40,  -7.96, -11.34>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.04,  -8.30, -12.11>, <  0.40,  -7.96, -11.34>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  0.75,  -7.62, -10.56>, <  1.47,  -8.07, -10.25>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.20,  -8.51,  -9.94>, <  1.47,  -8.07, -10.25>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -2.22,   0.50,  -4.16>, < -3.10,   0.62,  -4.26>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.98,   0.73,  -4.37>, < -3.10,   0.62,  -4.26>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -2.22,   0.50,  -4.16>, < -1.95,   0.92,  -4.41>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.68,   1.34,  -4.65>, < -1.95,   0.92,  -4.41>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.22,   0.50,  -4.16>, < -2.12,   0.46,  -3.61>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.01,   0.41,  -3.07>, < -2.12,   0.46,  -3.61>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.22,   0.50,  -4.16>, < -2.00,  -0.30,  -4.47>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.79,  -1.10,  -4.77>, < -2.00,  -0.30,  -4.47>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.83,  -0.48, -11.59>, <  4.11,  -0.05, -11.79>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.39,   0.38, -11.99>, <  4.11,  -0.05, -11.79>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.83,  -0.48, -11.59>, <  3.02,  -0.19, -11.26>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.21,   0.10, -10.94>, <  3.02,  -0.19, -11.26>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.83,  -0.48, -11.59>, <  3.77,  -0.89, -11.94>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.70,  -1.30, -12.30>, <  3.77,  -0.89, -11.94>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.83,  -0.48, -11.59>, <  4.30,  -0.83, -10.91>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.77,  -1.18, -10.24>, <  4.30,  -0.83, -10.91>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  7.00,  -1.74,  -4.56>, <  6.73,  -1.40,  -4.22>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  6.46,  -1.06,  -3.87>, <  6.73,  -1.40,  -4.22>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  7.00,  -1.74,  -4.56>, <  7.84,  -1.45,  -4.60>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  8.68,  -1.16,  -4.63>, <  7.84,  -1.45,  -4.60>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  7.00,  -1.74,  -4.56>, <  6.95,  -2.57,  -4.17>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  6.91,  -3.41,  -3.78>, <  6.95,  -2.57,  -4.17>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  7.00,  -1.74,  -4.56>, <  6.77,  -1.73,  -5.06>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  6.54,  -1.72,  -5.56>, <  6.77,  -1.73,  -5.06>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
// no constraints
