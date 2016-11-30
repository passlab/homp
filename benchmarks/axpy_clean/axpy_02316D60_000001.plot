set title "Offloading (axpy_kernel) Profile on 6 Devices"
set yrange [0:72.000000]
set xlabel "execution time in ms"
set xrange [0:158.400000]
set style fill pattern 2 bo 1
set style rect fs solid 1 noborder
set border 15 lw 0.2
set xtics out nomirror
unset key
set ytics out nomirror ("dev 0(sysid:0,type:HOSTCPU)" 5,"dev 1(sysid:1,type:HOSTCPU)" 15,"dev 2(sysid:0,type:THSIM)" 25,"dev 3(sysid:1,type:THSIM)" 35,"dev 4(sysid:2,type:THSIM)" 45,"dev 5(sysid:3,type:THSIM)" 55)
set object 1 rect from 4, 65 to 17, 68 fc rgb "#FF0000"
set label "ACCU_TOTAL" at 4,63 font "Helvetica,8'"

set object 2 rect from 21, 65 to 34, 68 fc rgb "#00FF00"
set label "INIT_0" at 21,63 font "Helvetica,8'"

set object 3 rect from 38, 65 to 51, 68 fc rgb "#0000FF"
set label "INIT_0.1" at 38,63 font "Helvetica,8'"

set object 4 rect from 55, 65 to 68, 68 fc rgb "#FFFF00"
set label "INIT_1" at 55,63 font "Helvetica,8'"

set object 5 rect from 72, 65 to 85, 68 fc rgb "#00FFFF"
set label "MODELING" at 72,63 font "Helvetica,8'"

set object 6 rect from 89, 65 to 102, 68 fc rgb "#FF00FF"
set label "ACC_MAPTO" at 89,63 font "Helvetica,8'"

set object 7 rect from 106, 65 to 119, 68 fc rgb "#808080"
set label "KERN" at 106,63 font "Helvetica,8'"

set object 8 rect from 123, 65 to 136, 68 fc rgb "#800000"
set label "PRE_BAR_X" at 123,63 font "Helvetica,8'"

set object 9 rect from 140, 65 to 153, 68 fc rgb "#808000"
set label "DATA_X" at 140,63 font "Helvetica,8'"

set object 10 rect from 157, 65 to 170, 68 fc rgb "#008000"
set label "POST_BAR_X" at 157,63 font "Helvetica,8'"

set object 11 rect from 174, 65 to 187, 68 fc rgb "#800080"
set label "ACC_MAPFROM" at 174,63 font "Helvetica,8'"

set object 12 rect from 191, 65 to 204, 68 fc rgb "#008080"
set label "FINI_1" at 191,63 font "Helvetica,8'"

set object 13 rect from 208, 65 to 221, 68 fc rgb "#000080"
set label "BAR_FINI_2" at 208,63 font "Helvetica,8'"

set object 14 rect from 225, 65 to 238, 68 fc rgb "(null)"
set label "PROF_BAR" at 225,63 font "Helvetica,8'"

set object 15 rect from 0.206185, 0 to 157.539293, 10 fc rgb "#FF0000"
set object 16 rect from 0.127973, 0 to 94.392098, 10 fc rgb "#00FF00"
set object 17 rect from 0.130016, 0 to 94.868206, 10 fc rgb "#0000FF"
set object 18 rect from 0.130422, 0 to 95.343587, 10 fc rgb "#FFFF00"
set object 19 rect from 0.131084, 0 to 96.068679, 10 fc rgb "#FF00FF"
set object 20 rect from 0.132089, 0 to 101.966911, 10 fc rgb "#808080"
set object 21 rect from 0.140179, 0 to 102.327271, 10 fc rgb "#800080"
set object 22 rect from 0.140901, 0 to 102.699270, 10 fc rgb "#008080"
set object 23 rect from 0.141180, 0 to 149.539337, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.199946, 10 to 155.275218, 20 fc rgb "#FF0000"
set object 25 rect from 0.041150, 10 to 31.929954, 20 fc rgb "#00FF00"
set object 26 rect from 0.044288, 10 to 32.963704, 20 fc rgb "#0000FF"
set object 27 rect from 0.045406, 10 to 34.260268, 20 fc rgb "#FFFF00"
set object 28 rect from 0.047232, 10 to 35.409776, 20 fc rgb "#FF00FF"
set object 29 rect from 0.048761, 10 to 41.835081, 20 fc rgb "#808080"
set object 30 rect from 0.057723, 10 to 42.346857, 20 fc rgb "#800080"
set object 31 rect from 0.058861, 10 to 42.999147, 20 fc rgb "#008080"
set object 32 rect from 0.059187, 10 to 144.793520, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.208525, 20 to 163.817531, 30 fc rgb "#FF0000"
set object 34 rect from 0.148259, 20 to 108.899630, 30 fc rgb "#00FF00"
set object 35 rect from 0.149934, 20 to 109.318949, 30 fc rgb "#0000FF"
set object 36 rect from 0.150281, 20 to 111.485468, 30 fc rgb "#FFFF00"
set object 37 rect from 0.153290, 20 to 115.169862, 30 fc rgb "#FF00FF"
set object 38 rect from 0.158309, 20 to 120.990206, 30 fc rgb "#808080"
set object 39 rect from 0.166307, 20 to 121.432826, 30 fc rgb "#800080"
set object 40 rect from 0.167149, 20 to 121.905289, 30 fc rgb "#008080"
set object 41 rect from 0.167563, 20 to 151.277067, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.210322, 30 to 164.442159, 40 fc rgb "#FF0000"
set object 43 rect from 0.175343, 30 to 128.532246, 40 fc rgb "#00FF00"
set object 44 rect from 0.176882, 30 to 128.945750, 40 fc rgb "#0000FF"
set object 45 rect from 0.177232, 30 to 130.786863, 40 fc rgb "#FFFF00"
set object 46 rect from 0.179790, 30 to 134.122536, 40 fc rgb "#FF00FF"
set object 47 rect from 0.184368, 30 to 139.920305, 40 fc rgb "#808080"
set object 48 rect from 0.192312, 30 to 140.372396, 40 fc rgb "#800080"
set object 49 rect from 0.193177, 30 to 140.808464, 40 fc rgb "#008080"
set object 50 rect from 0.193546, 30 to 152.687188, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.202034, 40 to 160.009405, 50 fc rgb "#FF0000"
set object 52 rect from 0.068722, 40 to 51.315052, 50 fc rgb "#00FF00"
set object 53 rect from 0.070827, 40 to 51.905465, 50 fc rgb "#0000FF"
set object 54 rect from 0.071407, 40 to 54.064706, 50 fc rgb "#FFFF00"
set object 55 rect from 0.074408, 40 to 58.393373, 50 fc rgb "#FF00FF"
set object 56 rect from 0.080328, 40 to 64.245013, 50 fc rgb "#808080"
set object 57 rect from 0.088367, 40 to 64.757527, 50 fc rgb "#800080"
set object 58 rect from 0.089351, 40 to 65.524101, 50 fc rgb "#008080"
set object 59 rect from 0.090126, 40 to 146.503589, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.204120, 50 to 160.853879, 60 fc rgb "#FF0000"
set object 61 rect from 0.097574, 50 to 72.069525, 60 fc rgb "#00FF00"
set object 62 rect from 0.099328, 50 to 72.563836, 60 fc rgb "#0000FF"
set object 63 rect from 0.099783, 50 to 74.603683, 60 fc rgb "#FFFF00"
set object 64 rect from 0.102611, 50 to 78.448237, 60 fc rgb "#FF00FF"
set object 65 rect from 0.107891, 50 to 84.399613, 60 fc rgb "#808080"
set object 66 rect from 0.116045, 50 to 84.833501, 60 fc rgb "#800080"
set object 67 rect from 0.116881, 50 to 85.297231, 60 fc rgb "#008080"
set object 68 rect from 0.117277, 50 to 147.992336, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
