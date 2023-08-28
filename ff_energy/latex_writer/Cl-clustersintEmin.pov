#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic angle 0
  right -9.65*x up 14.76*y
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
atom(<  0.25,  -6.13,  -3.10>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #0
atom(<  0.73,  -6.91,  -3.31>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #1
atom(< -0.61,  -6.28,  -3.48>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #2
atom(< -1.38,  -1.55,  -4.38>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #3
atom(< -1.81,  -0.82,  -3.88>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #4
atom(< -2.01,  -1.58,  -5.08>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #5
atom(< -3.47,  -2.27,  -0.52>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #6
atom(< -4.35,  -2.37,  -0.15>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #7
atom(< -3.53,  -2.73,  -1.39>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #8
atom(< -4.33,  -2.94,  -4.06>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #9
atom(< -3.55,  -3.45,  -4.47>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #10
atom(< -3.98,  -2.11,  -3.63>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #11
atom(<  3.20,  -4.29,  -4.09>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #12
atom(<  2.37,  -4.09,  -4.55>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #13
atom(<  3.94,  -4.14,  -4.72>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #14
atom(<  0.93,  -3.90,  -5.69>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #15
atom(<  1.53,  -3.87,  -6.49>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #16
atom(<  0.01,  -3.85,  -6.04>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #17
atom(<  3.83,  -5.07,  -1.61>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #18
atom(<  4.47,  -4.36,  -1.69>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #19
atom(<  3.34,  -5.04,  -2.47>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #20
atom(< -0.59,  -2.41,  -0.83>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #21
atom(< -0.32,  -1.97,   0.00>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #22
atom(< -1.55,  -2.34,  -0.69>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #23
atom(<  1.08,  -0.77,  -5.12>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #24
atom(<  0.24,  -1.28,  -5.01>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #25
atom(<  1.69,  -1.07,  -4.43>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #26
atom(<  2.24,  -1.88,  -2.55>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #27
atom(<  3.18,  -2.08,  -2.50>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #28
atom(<  2.07,  -1.17,  -1.84>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #29
atom(< -2.18,  -4.48,  -4.38>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #30
atom(< -2.34,  -5.49,  -4.46>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #31
atom(< -1.92,  -4.34,  -5.33>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #32
atom(<  1.43,  -4.12,  -0.62>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #33
atom(<  2.25,  -4.11,  -1.07>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #34
atom(<  1.16,  -3.16,  -0.74>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #35
atom(<  0.03,  -3.55,  -3.06>, 0.81, rgb <0.50, 0.20, 0.39>, 0.0, jmol) // #36
cylinder {<  0.25,  -6.13,  -3.10>, < -0.18,  -6.21,  -3.29>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.61,  -6.28,  -3.48>, < -0.18,  -6.21,  -3.29>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.25,  -6.13,  -3.10>, <  0.14,  -4.84,  -3.08>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.03,  -3.55,  -3.06>, <  0.14,  -4.84,  -3.08>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  0.25,  -6.13,  -3.10>, <  0.49,  -6.52,  -3.21>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.73,  -6.91,  -3.31>, <  0.49,  -6.52,  -3.21>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.61,  -6.28,  -3.48>, < -0.29,  -4.92,  -3.27>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.03,  -3.55,  -3.06>, < -0.29,  -4.92,  -3.27>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {< -1.38,  -1.55,  -4.38>, < -1.70,  -1.56,  -4.73>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.01,  -1.58,  -5.08>, < -1.70,  -1.56,  -4.73>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.38,  -1.55,  -4.38>, < -1.60,  -1.19,  -4.13>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.81,  -0.82,  -3.88>, < -1.60,  -1.19,  -4.13>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.38,  -1.55,  -4.38>, < -0.68,  -2.55,  -3.72>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.03,  -3.55,  -3.06>, < -0.68,  -2.55,  -3.72>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {< -3.47,  -2.27,  -0.52>, < -3.50,  -2.50,  -0.95>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.53,  -2.73,  -1.39>, < -3.50,  -2.50,  -0.95>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.47,  -2.27,  -0.52>, < -3.91,  -2.32,  -0.33>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -4.35,  -2.37,  -0.15>, < -3.91,  -2.32,  -0.33>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.33,  -2.94,  -4.06>, < -3.94,  -3.20,  -4.27>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.55,  -3.45,  -4.47>, < -3.94,  -3.20,  -4.27>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.33,  -2.94,  -4.06>, < -4.16,  -2.53,  -3.85>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.98,  -2.11,  -3.63>, < -4.16,  -2.53,  -3.85>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.20,  -4.29,  -4.09>, <  1.61,  -3.92,  -3.58>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.03,  -3.55,  -3.06>, <  1.61,  -3.92,  -3.58>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  3.20,  -4.29,  -4.09>, <  3.57,  -4.22,  -4.41>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.94,  -4.14,  -4.72>, <  3.57,  -4.22,  -4.41>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.20,  -4.29,  -4.09>, <  2.78,  -4.19,  -4.32>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.37,  -4.09,  -4.55>, <  2.78,  -4.19,  -4.32>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.37,  -4.09,  -4.55>, <  1.20,  -3.82,  -3.80>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.03,  -3.55,  -3.06>, <  1.20,  -3.82,  -3.80>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  0.93,  -3.90,  -5.69>, <  0.47,  -3.88,  -5.86>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.01,  -3.85,  -6.04>, <  0.47,  -3.88,  -5.86>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.93,  -3.90,  -5.69>, <  0.48,  -3.73,  -4.37>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.03,  -3.55,  -3.06>, <  0.48,  -3.73,  -4.37>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  0.93,  -3.90,  -5.69>, <  1.23,  -3.89,  -6.09>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.53,  -3.87,  -6.49>, <  1.23,  -3.89,  -6.09>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.01,  -3.85,  -6.04>, <  0.02,  -3.70,  -4.55>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.03,  -3.55,  -3.06>, <  0.02,  -3.70,  -4.55>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  3.83,  -5.07,  -1.61>, <  3.59,  -5.05,  -2.04>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.34,  -5.04,  -2.47>, <  3.59,  -5.05,  -2.04>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.83,  -5.07,  -1.61>, <  4.15,  -4.71,  -1.65>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.47,  -4.36,  -1.69>, <  4.15,  -4.71,  -1.65>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.59,  -2.41,  -0.83>, < -0.28,  -2.98,  -1.95>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.03,  -3.55,  -3.06>, < -0.28,  -2.98,  -1.95>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {< -0.59,  -2.41,  -0.83>, < -1.07,  -2.38,  -0.76>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.55,  -2.34,  -0.69>, < -1.07,  -2.38,  -0.76>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.59,  -2.41,  -0.83>, < -0.45,  -2.19,  -0.42>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.32,  -1.97,   0.00>, < -0.45,  -2.19,  -0.42>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.55,  -2.34,  -0.69>, < -0.76,  -2.95,  -1.88>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.03,  -3.55,  -3.06>, < -0.76,  -2.95,  -1.88>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  1.08,  -0.77,  -5.12>, <  1.38,  -0.92,  -4.77>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.69,  -1.07,  -4.43>, <  1.38,  -0.92,  -4.77>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.08,  -0.77,  -5.12>, <  0.66,  -1.03,  -5.07>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.24,  -1.28,  -5.01>, <  0.66,  -1.03,  -5.07>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.24,  -1.28,  -5.01>, <  0.13,  -2.42,  -4.04>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.03,  -3.55,  -3.06>, <  0.13,  -2.42,  -4.04>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  2.24,  -1.88,  -2.55>, <  1.13,  -2.71,  -2.80>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.03,  -3.55,  -3.06>, <  1.13,  -2.71,  -2.80>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  2.24,  -1.88,  -2.55>, <  2.15,  -1.53,  -2.19>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.07,  -1.17,  -1.84>, <  2.15,  -1.53,  -2.19>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.24,  -1.88,  -2.55>, <  2.71,  -1.98,  -2.52>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.18,  -2.08,  -2.50>, <  2.71,  -1.98,  -2.52>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.18,  -4.48,  -4.38>, < -2.26,  -4.99,  -4.42>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.34,  -5.49,  -4.46>, < -2.26,  -4.99,  -4.42>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.18,  -4.48,  -4.38>, < -2.05,  -4.41,  -4.85>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.92,  -4.34,  -5.33>, < -2.05,  -4.41,  -4.85>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.18,  -4.48,  -4.38>, < -1.07,  -4.02,  -3.72>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.03,  -3.55,  -3.06>, < -1.07,  -4.02,  -3.72>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {< -1.92,  -4.34,  -5.33>, < -0.95,  -3.94,  -4.19>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.03,  -3.55,  -3.06>, < -0.95,  -3.94,  -4.19>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  1.43,  -4.12,  -0.62>, <  0.73,  -3.83,  -1.84>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.03,  -3.55,  -3.06>, <  0.73,  -3.83,  -1.84>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  1.43,  -4.12,  -0.62>, <  1.84,  -4.11,  -0.84>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.25,  -4.11,  -1.07>, <  1.84,  -4.11,  -0.84>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.43,  -4.12,  -0.62>, <  1.29,  -3.64,  -0.68>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.16,  -3.16,  -0.74>, <  1.29,  -3.64,  -0.68>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.25,  -4.11,  -1.07>, <  1.14,  -3.83,  -2.06>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.03,  -3.55,  -3.06>, <  1.14,  -3.83,  -2.06>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  1.16,  -3.16,  -0.74>, <  0.59,  -3.35,  -1.90>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.03,  -3.55,  -3.06>, <  0.59,  -3.35,  -1.90>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
// no constraints
