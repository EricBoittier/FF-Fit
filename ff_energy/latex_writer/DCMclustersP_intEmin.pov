#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic angle 0
  right -21.93*x up 19.66*y
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
atom(< -2.80,   7.88,  -4.31>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #0
atom(< -4.30,   8.91,  -4.45>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #1
atom(< -2.86,   6.68,  -5.63>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #2
atom(< -2.72,   7.29,  -3.37>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #3
atom(< -1.88,   8.51,  -4.45>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #4
atom(< -6.11,   2.33, -12.04>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #5
atom(< -6.59,   2.53, -13.71>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #6
atom(< -4.58,   1.40, -11.95>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #7
atom(< -6.89,   1.77, -11.55>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #8
atom(< -5.95,   3.33, -11.60>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #9
atom(< -1.03,   4.13,  -2.97>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #10
atom(< -0.42,   5.76,  -2.85>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #11
atom(< -1.52,   3.79,  -4.63>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #12
atom(< -1.86,   3.94,  -2.28>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #13
atom(< -0.12,   3.54,  -2.86>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #14
atom(< -2.57,  -1.38,  -6.16>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #15
atom(< -3.70,  -0.58,  -5.10>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #16
atom(< -2.21,  -0.18,  -7.43>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #17
atom(< -3.11,  -2.13,  -6.54>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #18
atom(< -1.73,  -1.60,  -5.46>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #19
atom(< -7.19,   0.15,  -7.04>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #20
atom(< -7.36,   0.91,  -8.62>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #21
atom(< -7.24,  -1.63,  -7.17>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #22
atom(< -8.05,   0.44,  -6.38>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #23
atom(< -6.23,   0.43,  -6.61>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #24
atom(< -7.56,   6.41, -15.52>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #25
atom(< -7.30,   8.16, -15.65>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #26
atom(< -7.40,   5.51, -17.03>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #27
atom(< -8.65,   6.29, -15.19>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #28
atom(< -6.80,   5.95, -14.84>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #29
atom(<  3.59,   2.01, -16.24>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #30
atom(<  1.99,   1.90, -15.42>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #31
atom(<  4.37,   3.51, -15.75>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #32
atom(<  3.29,   2.06, -17.29>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #33
atom(<  4.18,   1.08, -16.03>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #34
atom(<  2.93,   4.94,  -6.39>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #35
atom(<  2.32,   6.51,  -6.89>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #36
atom(<  2.57,   4.75,  -4.68>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #37
atom(<  4.03,   4.96,  -6.48>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #38
atom(<  2.55,   4.13,  -7.07>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #39
atom(<  1.70,   5.12, -13.31>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #40
atom(< -0.01,   4.97, -13.89>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #41
atom(<  2.42,   6.78, -13.44>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #42
atom(<  2.32,   4.42, -13.96>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #43
atom(<  1.81,   4.87, -12.25>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #44
atom(< -9.39,   6.58, -12.12>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #45
atom(< -8.16,   5.38, -12.60>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #46
atom(< -8.77,   8.17, -12.61>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #47
atom(< -9.47,   6.54, -11.00>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #48
atom(<-10.32,   6.35, -12.64>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #49
atom(< -5.11,   8.12,  -9.13>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #50
atom(< -4.51,   6.48,  -8.72>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #51
atom(< -6.74,   7.96,  -9.79>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #52
atom(< -4.44,   8.60,  -9.89>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #53
atom(< -5.10,   8.74,  -8.22>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #54
atom(<  3.82,   1.53,  -5.08>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #55
atom(<  4.32,  -0.24,  -5.18>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #56
atom(<  4.13,   2.19,  -6.69>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #57
atom(<  4.47,   2.03,  -4.27>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #58
atom(<  2.74,   1.62,  -4.77>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #59
atom(< -1.94,  -4.97,  -9.24>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #60
atom(< -1.03,  -5.62,  -7.79>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #61
atom(< -2.24,  -3.25,  -8.92>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #62
atom(< -2.91,  -5.54,  -9.32>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #63
atom(< -1.35,  -5.17, -10.19>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #64
atom(< -3.44,   6.26, -12.16>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #65
atom(< -2.09,   6.87, -11.08>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #66
atom(< -4.65,   7.49, -12.80>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #67
atom(< -4.01,   5.41, -11.60>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #68
atom(< -3.07,   5.87, -13.09>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #69
atom(<  8.27,   6.82, -13.87>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #70
atom(< 10.04,   6.95, -13.56>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #71
atom(<  7.83,   7.21, -15.55>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #72
atom(<  7.68,   7.44, -13.18>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #73
atom(<  7.95,   5.77, -13.70>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #74
atom(< -0.78,   0.29, -13.92>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #75
atom(< -1.75,   0.27, -15.37>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #76
atom(< -0.57,  -1.41, -13.40>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #77
atom(< -1.35,   0.78, -13.10>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #78
atom(<  0.13,   0.84, -14.15>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #79
atom(< -2.93,   8.36, -16.23>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #80
atom(< -1.80,   8.95, -15.00>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #81
atom(< -3.19,   6.59, -16.10>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #82
atom(< -2.53,   8.59, -17.27>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #83
atom(< -3.92,   8.85, -16.14>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #84
atom(<  5.57,  -2.37, -12.80>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #85
atom(<  3.90,  -2.21, -13.44>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #86
atom(<  6.60,  -1.11, -13.55>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #87
atom(<  5.98,  -3.35, -13.15>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #88
atom(<  5.61,  -2.23, -11.72>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #89
atom(<  6.06,   2.21, -12.17>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #90
atom(<  5.55,   3.93, -12.51>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #91
atom(<  5.73,   1.70, -10.45>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #92
atom(<  7.12,   2.11, -12.38>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #93
atom(<  5.45,   1.58, -12.83>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #94
atom(<  6.05,   6.04,  -9.15>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #95
atom(<  4.63,   5.79, -10.19>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #96
atom(<  7.52,   5.27,  -9.86>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #97
atom(<  5.82,   5.50,  -8.22>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #98
atom(<  6.02,   7.16,  -9.09>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #99
cylinder {< -2.80,   7.88,  -4.31>, < -2.83,   7.28,  -4.97>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.86,   6.68,  -5.63>, < -2.83,   7.28,  -4.97>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -2.80,   7.88,  -4.31>, < -3.55,   8.39,  -4.38>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.30,   8.91,  -4.45>, < -3.55,   8.39,  -4.38>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -2.80,   7.88,  -4.31>, < -2.34,   8.19,  -4.38>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.88,   8.51,  -4.45>, < -2.34,   8.19,  -4.38>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.80,   7.88,  -4.31>, < -2.76,   7.59,  -3.84>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.72,   7.29,  -3.37>, < -2.76,   7.59,  -3.84>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -6.11,   2.33, -12.04>, < -6.35,   2.43, -12.87>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.59,   2.53, -13.71>, < -6.35,   2.43, -12.87>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -6.11,   2.33, -12.04>, < -6.50,   2.05, -11.79>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.89,   1.77, -11.55>, < -6.50,   2.05, -11.79>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -6.11,   2.33, -12.04>, < -6.03,   2.83, -11.82>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -5.95,   3.33, -11.60>, < -6.03,   2.83, -11.82>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -6.11,   2.33, -12.04>, < -5.35,   1.87, -11.99>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.58,   1.40, -11.95>, < -5.35,   1.87, -11.99>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.03,   4.13,  -2.97>, < -1.44,   4.03,  -2.62>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.86,   3.94,  -2.28>, < -1.44,   4.03,  -2.62>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.03,   4.13,  -2.97>, < -0.72,   4.94,  -2.91>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.42,   5.76,  -2.85>, < -0.72,   4.94,  -2.91>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.03,   4.13,  -2.97>, < -1.28,   3.96,  -3.80>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.52,   3.79,  -4.63>, < -1.28,   3.96,  -3.80>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.03,   4.13,  -2.97>, < -0.58,   3.83,  -2.91>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.12,   3.54,  -2.86>, < -0.58,   3.83,  -2.91>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.57,  -1.38,  -6.16>, < -3.14,  -0.98,  -5.63>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.70,  -0.58,  -5.10>, < -3.14,  -0.98,  -5.63>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -2.57,  -1.38,  -6.16>, < -2.39,  -0.78,  -6.79>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.21,  -0.18,  -7.43>, < -2.39,  -0.78,  -6.79>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -2.57,  -1.38,  -6.16>, < -2.84,  -1.76,  -6.35>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.11,  -2.13,  -6.54>, < -2.84,  -1.76,  -6.35>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.57,  -1.38,  -6.16>, < -2.15,  -1.49,  -5.81>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.73,  -1.60,  -5.46>, < -2.15,  -1.49,  -5.81>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -7.19,   0.15,  -7.04>, < -7.27,   0.53,  -7.83>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -7.36,   0.91,  -8.62>, < -7.27,   0.53,  -7.83>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -7.19,   0.15,  -7.04>, < -7.62,   0.29,  -6.71>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -8.05,   0.44,  -6.38>, < -7.62,   0.29,  -6.71>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -7.19,   0.15,  -7.04>, < -6.71,   0.29,  -6.82>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.23,   0.43,  -6.61>, < -6.71,   0.29,  -6.82>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -7.19,   0.15,  -7.04>, < -7.22,  -0.74,  -7.11>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -7.24,  -1.63,  -7.17>, < -7.22,  -0.74,  -7.11>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -7.56,   6.41, -15.52>, < -7.43,   7.29, -15.58>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -7.30,   8.16, -15.65>, < -7.43,   7.29, -15.58>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -7.56,   6.41, -15.52>, < -8.10,   6.35, -15.35>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -8.65,   6.29, -15.19>, < -8.10,   6.35, -15.35>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -7.56,   6.41, -15.52>, < -7.18,   6.18, -15.18>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.80,   5.95, -14.84>, < -7.18,   6.18, -15.18>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -7.56,   6.41, -15.52>, < -7.48,   5.96, -16.27>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -7.40,   5.51, -17.03>, < -7.48,   5.96, -16.27>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.59,   2.01, -16.24>, <  2.79,   1.95, -15.83>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.99,   1.90, -15.42>, <  2.79,   1.95, -15.83>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.59,   2.01, -16.24>, <  3.44,   2.04, -16.77>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.29,   2.06, -17.29>, <  3.44,   2.04, -16.77>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.59,   2.01, -16.24>, <  3.89,   1.54, -16.13>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.18,   1.08, -16.03>, <  3.89,   1.54, -16.13>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.59,   2.01, -16.24>, <  3.98,   2.76, -16.00>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.37,   3.51, -15.75>, <  3.98,   2.76, -16.00>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  2.93,   4.94,  -6.39>, <  2.63,   5.73,  -6.64>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.32,   6.51,  -6.89>, <  2.63,   5.73,  -6.64>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  2.93,   4.94,  -6.39>, <  3.48,   4.95,  -6.44>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.03,   4.96,  -6.48>, <  3.48,   4.95,  -6.44>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.93,   4.94,  -6.39>, <  2.75,   4.85,  -5.53>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.57,   4.75,  -4.68>, <  2.75,   4.85,  -5.53>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  2.93,   4.94,  -6.39>, <  2.74,   4.54,  -6.73>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.55,   4.13,  -7.07>, <  2.74,   4.54,  -6.73>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.70,   5.12, -13.31>, <  0.84,   5.05, -13.60>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.01,   4.97, -13.89>, <  0.84,   5.05, -13.60>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  1.70,   5.12, -13.31>, <  2.06,   5.95, -13.38>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.42,   6.78, -13.44>, <  2.06,   5.95, -13.38>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  1.70,   5.12, -13.31>, <  2.01,   4.77, -13.64>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.32,   4.42, -13.96>, <  2.01,   4.77, -13.64>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.70,   5.12, -13.31>, <  1.75,   5.00, -12.78>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.81,   4.87, -12.25>, <  1.75,   5.00, -12.78>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -9.39,   6.58, -12.12>, < -9.08,   7.38, -12.37>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -8.77,   8.17, -12.61>, < -9.08,   7.38, -12.37>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -9.39,   6.58, -12.12>, < -9.86,   6.46, -12.38>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<-10.32,   6.35, -12.64>, < -9.86,   6.46, -12.38>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -9.39,   6.58, -12.12>, < -8.78,   5.98, -12.36>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -8.16,   5.38, -12.60>, < -8.78,   5.98, -12.36>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -9.39,   6.58, -12.12>, < -9.43,   6.56, -11.56>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -9.47,   6.54, -11.00>, < -9.43,   6.56, -11.56>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -5.11,   8.12,  -9.13>, < -4.77,   8.36,  -9.51>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.44,   8.60,  -9.89>, < -4.77,   8.36,  -9.51>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -5.11,   8.12,  -9.13>, < -5.92,   8.04,  -9.46>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.74,   7.96,  -9.79>, < -5.92,   8.04,  -9.46>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -5.11,   8.12,  -9.13>, < -5.10,   8.43,  -8.67>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -5.10,   8.74,  -8.22>, < -5.10,   8.43,  -8.67>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -5.11,   8.12,  -9.13>, < -4.81,   7.30,  -8.93>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.51,   6.48,  -8.72>, < -4.81,   7.30,  -8.93>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.82,   1.53,  -5.08>, <  4.15,   1.78,  -4.68>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.47,   2.03,  -4.27>, <  4.15,   1.78,  -4.68>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.82,   1.53,  -5.08>, <  3.98,   1.86,  -5.88>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.13,   2.19,  -6.69>, <  3.98,   1.86,  -5.88>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.82,   1.53,  -5.08>, <  3.28,   1.58,  -4.92>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.74,   1.62,  -4.77>, <  3.28,   1.58,  -4.92>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.82,   1.53,  -5.08>, <  4.07,   0.65,  -5.13>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.32,  -0.24,  -5.18>, <  4.07,   0.65,  -5.13>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.94,  -4.97,  -9.24>, < -2.09,  -4.11,  -9.08>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.24,  -3.25,  -8.92>, < -2.09,  -4.11,  -9.08>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.94,  -4.97,  -9.24>, < -2.42,  -5.25,  -9.28>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.91,  -5.54,  -9.32>, < -2.42,  -5.25,  -9.28>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.94,  -4.97,  -9.24>, < -1.48,  -5.29,  -8.52>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.03,  -5.62,  -7.79>, < -1.48,  -5.29,  -8.52>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.94,  -4.97,  -9.24>, < -1.64,  -5.07,  -9.71>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.35,  -5.17, -10.19>, < -1.64,  -5.07,  -9.71>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.44,   6.26, -12.16>, < -2.76,   6.57, -11.62>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.09,   6.87, -11.08>, < -2.76,   6.57, -11.62>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -3.44,   6.26, -12.16>, < -4.04,   6.88, -12.48>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.65,   7.49, -12.80>, < -4.04,   6.88, -12.48>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -3.44,   6.26, -12.16>, < -3.73,   5.84, -11.88>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.01,   5.41, -11.60>, < -3.73,   5.84, -11.88>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.44,   6.26, -12.16>, < -3.26,   6.07, -12.62>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.07,   5.87, -13.09>, < -3.26,   6.07, -12.62>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  8.27,   6.82, -13.87>, <  8.05,   7.01, -14.71>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  7.83,   7.21, -15.55>, <  8.05,   7.01, -14.71>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  8.27,   6.82, -13.87>, <  7.98,   7.13, -13.53>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  7.68,   7.44, -13.18>, <  7.98,   7.13, -13.53>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  8.27,   6.82, -13.87>, <  8.11,   6.29, -13.78>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  7.95,   5.77, -13.70>, <  8.11,   6.29, -13.78>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  8.27,   6.82, -13.87>, <  9.15,   6.88, -13.71>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< 10.04,   6.95, -13.56>, <  9.15,   6.88, -13.71>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -0.78,   0.29, -13.92>, < -1.27,   0.28, -14.65>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.75,   0.27, -15.37>, < -1.27,   0.28, -14.65>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -0.78,   0.29, -13.92>, < -0.67,  -0.56, -13.66>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.57,  -1.41, -13.40>, < -0.67,  -0.56, -13.66>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -0.78,   0.29, -13.92>, < -1.07,   0.53, -13.51>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.35,   0.78, -13.10>, < -1.07,   0.53, -13.51>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.78,   0.29, -13.92>, < -0.32,   0.56, -14.04>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.13,   0.84, -14.15>, < -0.32,   0.56, -14.04>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.93,   8.36, -16.23>, < -2.36,   8.66, -15.62>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.80,   8.95, -15.00>, < -2.36,   8.66, -15.62>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -2.93,   8.36, -16.23>, < -2.73,   8.47, -16.75>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.53,   8.59, -17.27>, < -2.73,   8.47, -16.75>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.93,   8.36, -16.23>, < -3.42,   8.60, -16.19>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.92,   8.85, -16.14>, < -3.42,   8.60, -16.19>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.93,   8.36, -16.23>, < -3.06,   7.47, -16.17>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.19,   6.59, -16.10>, < -3.06,   7.47, -16.17>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  5.57,  -2.37, -12.80>, <  4.73,  -2.29, -13.12>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.90,  -2.21, -13.44>, <  4.73,  -2.29, -13.12>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  5.57,  -2.37, -12.80>, <  5.78,  -2.86, -12.98>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.98,  -3.35, -13.15>, <  5.78,  -2.86, -12.98>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  5.57,  -2.37, -12.80>, <  6.08,  -1.74, -13.18>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  6.60,  -1.11, -13.55>, <  6.08,  -1.74, -13.18>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  5.57,  -2.37, -12.80>, <  5.59,  -2.30, -12.26>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.61,  -2.23, -11.72>, <  5.59,  -2.30, -12.26>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  6.06,   2.21, -12.17>, <  5.76,   1.90, -12.50>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.45,   1.58, -12.83>, <  5.76,   1.90, -12.50>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  6.06,   2.21, -12.17>, <  5.80,   3.07, -12.34>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.55,   3.93, -12.51>, <  5.80,   3.07, -12.34>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  6.06,   2.21, -12.17>, <  6.59,   2.16, -12.27>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  7.12,   2.11, -12.38>, <  6.59,   2.16, -12.27>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  6.06,   2.21, -12.17>, <  5.90,   1.96, -11.31>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.73,   1.70, -10.45>, <  5.90,   1.96, -11.31>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  6.05,   6.04,  -9.15>, <  6.78,   5.66,  -9.51>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  7.52,   5.27,  -9.86>, <  6.78,   5.66,  -9.51>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  6.05,   6.04,  -9.15>, <  5.94,   5.77,  -8.68>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.82,   5.50,  -8.22>, <  5.94,   5.77,  -8.68>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  6.05,   6.04,  -9.15>, <  5.34,   5.91,  -9.67>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.63,   5.79, -10.19>, <  5.34,   5.91,  -9.67>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  6.05,   6.04,  -9.15>, <  6.03,   6.60,  -9.12>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  6.02,   7.16,  -9.09>, <  6.03,   6.60,  -9.12>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
// no constraints
