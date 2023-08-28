#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic angle 0
  right -11.24*x up 7.62*y
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
atom(< -0.49,   2.34,  -6.83>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #0
atom(< -1.34,   2.22,  -7.27>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #1
atom(<  0.18,   2.00,  -7.49>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #2
atom(< -2.18,  -0.15,  -6.83>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #3
atom(< -2.46,   0.54,  -7.43>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #4
atom(< -2.90,  -0.43,  -6.30>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #5
atom(<  3.55,   2.23,  -6.81>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #6
atom(<  3.06,   1.42,  -6.50>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #7
atom(<  3.02,   2.97,  -6.46>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #8
atom(<  1.80,  -2.64,  -1.51>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #9
atom(<  1.39,  -2.57,  -2.37>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #10
atom(<  2.28,  -3.48,  -1.63>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #11
atom(< -1.15,  -3.12,  -5.19>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #12
atom(< -1.93,  -3.50,  -5.70>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #13
atom(< -1.50,  -2.43,  -4.63>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #14
atom(< -1.00,  -2.07,  -8.44>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #15
atom(< -1.90,  -2.52,  -8.52>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #16
atom(< -1.42,  -1.38,  -7.86>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #17
atom(<  2.42,   0.05,  -5.57>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #18
atom(<  2.86,  -0.06,  -4.66>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #19
atom(<  2.32,  -0.87,  -5.79>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #20
atom(< -2.27,  -1.21,  -3.77>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #21
atom(< -3.02,  -0.74,  -4.17>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #22
atom(< -2.41,  -1.14,  -2.77>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #23
atom(< -4.36,  -1.04,  -5.60>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #24
atom(< -5.23,  -0.55,  -5.76>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #25
atom(< -4.69,  -1.84,  -5.24>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #26
atom(<  1.07,   1.69,  -8.87>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #27
atom(<  0.92,   0.79,  -9.14>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #28
atom(<  1.95,   1.84,  -9.14>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #29
atom(<  1.46,  -2.86,  -4.42>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #30
atom(<  2.11,  -3.23,  -5.01>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #31
atom(<  0.65,  -3.26,  -4.75>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #32
atom(<  0.74,   0.97,  -2.61>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #33
atom(<  1.09,   0.45,  -1.91>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #34
atom(<  1.15,   1.87,  -2.42>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #35
atom(< -2.54,   2.04,  -4.37>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #36
atom(< -3.41,   1.83,  -4.83>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #37
atom(< -2.23,   2.94,  -4.70>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #38
atom(< -0.34,   0.09,  -5.11>, 0.81, rgb <0.50, 0.20, 0.39>, 0.0, jmol) // #39
cylinder {< -0.49,   2.34,  -6.83>, < -0.92,   2.28,  -7.05>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.34,   2.22,  -7.27>, < -0.92,   2.28,  -7.05>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.49,   2.34,  -6.83>, < -0.16,   2.17,  -7.16>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.18,   2.00,  -7.49>, < -0.16,   2.17,  -7.16>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.49,   2.34,  -6.83>, < -0.42,   1.22,  -5.97>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.34,   0.09,  -5.11>, < -0.42,   1.22,  -5.97>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  0.18,   2.00,  -7.49>, < -0.08,   1.04,  -6.30>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.34,   0.09,  -5.11>, < -0.08,   1.04,  -6.30>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {< -2.18,  -0.15,  -6.83>, < -2.32,   0.20,  -7.13>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.46,   0.54,  -7.43>, < -2.32,   0.20,  -7.13>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.18,  -0.15,  -6.83>, < -2.54,  -0.29,  -6.56>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.90,  -0.43,  -6.30>, < -2.54,  -0.29,  -6.56>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.18,  -0.15,  -6.83>, < -1.26,  -0.03,  -5.97>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.34,   0.09,  -5.11>, < -1.26,  -0.03,  -5.97>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {< -2.46,   0.54,  -7.43>, < -1.40,   0.32,  -6.27>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.34,   0.09,  -5.11>, < -1.40,   0.32,  -6.27>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {< -2.90,  -0.43,  -6.30>, < -1.62,  -0.17,  -5.70>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.34,   0.09,  -5.11>, < -1.62,  -0.17,  -5.70>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  3.55,   2.23,  -6.81>, <  3.31,   1.82,  -6.66>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.06,   1.42,  -6.50>, <  3.31,   1.82,  -6.66>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.55,   2.23,  -6.81>, <  3.29,   2.60,  -6.64>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.02,   2.97,  -6.46>, <  3.29,   2.60,  -6.64>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.80,  -2.64,  -1.51>, <  1.60,  -2.60,  -1.94>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.39,  -2.57,  -2.37>, <  1.60,  -2.60,  -1.94>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.80,  -2.64,  -1.51>, <  2.04,  -3.06,  -1.57>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.28,  -3.48,  -1.63>, <  2.04,  -3.06,  -1.57>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.15,  -3.12,  -5.19>, < -1.54,  -3.31,  -5.45>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.93,  -3.50,  -5.70>, < -1.54,  -3.31,  -5.45>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.15,  -3.12,  -5.19>, < -1.32,  -2.78,  -4.91>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.50,  -2.43,  -4.63>, < -1.32,  -2.78,  -4.91>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.15,  -3.12,  -5.19>, < -0.74,  -1.52,  -5.15>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.34,   0.09,  -5.11>, < -0.74,  -1.52,  -5.15>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {< -1.50,  -2.43,  -4.63>, < -0.92,  -1.17,  -4.87>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.34,   0.09,  -5.11>, < -0.92,  -1.17,  -4.87>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {< -1.00,  -2.07,  -8.44>, < -1.45,  -2.29,  -8.48>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.90,  -2.52,  -8.52>, < -1.45,  -2.29,  -8.48>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.00,  -2.07,  -8.44>, < -1.21,  -1.73,  -8.15>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.42,  -1.38,  -7.86>, < -1.21,  -1.73,  -8.15>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.42,   0.05,  -5.57>, <  2.37,  -0.41,  -5.68>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.32,  -0.87,  -5.79>, <  2.37,  -0.41,  -5.68>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.42,   0.05,  -5.57>, <  1.04,   0.07,  -5.34>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.34,   0.09,  -5.11>, <  1.04,   0.07,  -5.34>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  2.42,   0.05,  -5.57>, <  2.64,  -0.00,  -5.11>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.86,  -0.06,  -4.66>, <  2.64,  -0.00,  -5.11>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.32,  -0.87,  -5.79>, <  0.99,  -0.39,  -5.45>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.34,   0.09,  -5.11>, <  0.99,  -0.39,  -5.45>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {< -2.27,  -1.21,  -3.77>, < -2.64,  -0.97,  -3.97>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.02,  -0.74,  -4.17>, < -2.64,  -0.97,  -3.97>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.27,  -1.21,  -3.77>, < -2.34,  -1.17,  -3.27>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.41,  -1.14,  -2.77>, < -2.34,  -1.17,  -3.27>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.27,  -1.21,  -3.77>, < -1.30,  -0.56,  -4.44>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.34,   0.09,  -5.11>, < -1.30,  -0.56,  -4.44>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {< -3.02,  -0.74,  -4.17>, < -1.68,  -0.32,  -4.64>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.34,   0.09,  -5.11>, < -1.68,  -0.32,  -4.64>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {< -4.36,  -1.04,  -5.60>, < -4.80,  -0.79,  -5.68>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -5.23,  -0.55,  -5.76>, < -4.80,  -0.79,  -5.68>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.36,  -1.04,  -5.60>, < -4.52,  -1.44,  -5.42>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -4.69,  -1.84,  -5.24>, < -4.52,  -1.44,  -5.42>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.07,   1.69,  -8.87>, <  1.51,   1.77,  -9.00>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.95,   1.84,  -9.14>, <  1.51,   1.77,  -9.00>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.07,   1.69,  -8.87>, <  1.00,   1.24,  -9.01>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.92,   0.79,  -9.14>, <  1.00,   1.24,  -9.01>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.46,  -2.86,  -4.42>, <  0.56,  -1.39,  -4.77>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.34,   0.09,  -5.11>, <  0.56,  -1.39,  -4.77>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  1.46,  -2.86,  -4.42>, <  1.78,  -3.05,  -4.72>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.11,  -3.23,  -5.01>, <  1.78,  -3.05,  -4.72>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.46,  -2.86,  -4.42>, <  1.05,  -3.06,  -4.58>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.65,  -3.26,  -4.75>, <  1.05,  -3.06,  -4.58>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.74,   0.97,  -2.61>, <  0.20,   0.53,  -3.86>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.34,   0.09,  -5.11>, <  0.20,   0.53,  -3.86>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  0.74,   0.97,  -2.61>, <  0.94,   1.42,  -2.52>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.15,   1.87,  -2.42>, <  0.94,   1.42,  -2.52>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.74,   0.97,  -2.61>, <  0.91,   0.71,  -2.26>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.09,   0.45,  -1.91>, <  0.91,   0.71,  -2.26>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.54,   2.04,  -4.37>, < -2.39,   2.49,  -4.54>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.23,   2.94,  -4.70>, < -2.39,   2.49,  -4.54>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.54,   2.04,  -4.37>, < -2.98,   1.94,  -4.60>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.41,   1.83,  -4.83>, < -2.98,   1.94,  -4.60>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.54,   2.04,  -4.37>, < -1.44,   1.07,  -4.74>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.34,   0.09,  -5.11>, < -1.44,   1.07,  -4.74>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
// no constraints
