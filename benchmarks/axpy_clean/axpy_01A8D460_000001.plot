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

set object 15 rect from 0.239001, 0 to 158.241726, 10 fc rgb "#FF0000"
set object 16 rect from 0.138316, 0 to 86.977751, 10 fc rgb "#00FF00"
set object 17 rect from 0.140253, 0 to 87.387408, 10 fc rgb "#0000FF"
set object 18 rect from 0.140683, 0 to 87.834989, 10 fc rgb "#FFFF00"
set object 19 rect from 0.141405, 0 to 88.449167, 10 fc rgb "#FF00FF"
set object 20 rect from 0.142423, 0 to 96.731874, 10 fc rgb "#808080"
set object 21 rect from 0.155745, 0 to 97.095530, 10 fc rgb "#800080"
set object 22 rect from 0.156568, 0 to 97.433704, 10 fc rgb "#008080"
set object 23 rect from 0.156850, 0 to 148.201650, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.235816, 10 to 156.631062, 20 fc rgb "#FF0000"
set object 25 rect from 0.071132, 10 to 45.346017, 20 fc rgb "#00FF00"
set object 26 rect from 0.073288, 10 to 45.769352, 20 fc rgb "#0000FF"
set object 27 rect from 0.073737, 10 to 46.252368, 20 fc rgb "#FFFF00"
set object 28 rect from 0.074550, 10 to 47.021333, 20 fc rgb "#FF00FF"
set object 29 rect from 0.075774, 10 to 55.494259, 20 fc rgb "#808080"
set object 30 rect from 0.089444, 10 to 55.879054, 20 fc rgb "#800080"
set object 31 rect from 0.090258, 10 to 56.237120, 20 fc rgb "#008080"
set object 32 rect from 0.090607, 10 to 146.063214, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.233804, 20 to 163.132158, 30 fc rgb "#FF0000"
set object 34 rect from 0.028552, 20 to 19.447948, 30 fc rgb "#00FF00"
set object 35 rect from 0.031714, 20 to 20.233077, 30 fc rgb "#0000FF"
set object 36 rect from 0.032668, 20 to 22.816606, 30 fc rgb "#FFFF00"
set object 37 rect from 0.036863, 20 to 28.779362, 30 fc rgb "#FF00FF"
set object 38 rect from 0.046417, 20 to 37.134178, 30 fc rgb "#808080"
set object 39 rect from 0.059881, 20 to 37.769490, 30 fc rgb "#800080"
set object 40 rect from 0.061467, 20 to 38.555244, 30 fc rgb "#008080"
set object 41 rect from 0.062155, 20 to 144.762124, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.240429, 30 to 165.498726, 40 fc rgb "#FF0000"
set object 43 rect from 0.163372, 30 to 102.408052, 40 fc rgb "#00FF00"
set object 44 rect from 0.165063, 30 to 102.797196, 40 fc rgb "#0000FF"
set object 45 rect from 0.165473, 30 to 104.587514, 40 fc rgb "#FFFF00"
set object 46 rect from 0.168356, 30 to 110.181639, 40 fc rgb "#FF00FF"
set object 47 rect from 0.177364, 30 to 118.310180, 40 fc rgb "#808080"
set object 48 rect from 0.190446, 30 to 118.870895, 40 fc rgb "#800080"
set object 49 rect from 0.191594, 30 to 119.266880, 40 fc rgb "#008080"
set object 50 rect from 0.191986, 30 to 149.143432, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.237409, 40 to 164.010535, 50 fc rgb "#FF0000"
set object 52 rect from 0.098940, 40 to 62.432367, 50 fc rgb "#00FF00"
set object 53 rect from 0.100772, 40 to 62.902323, 50 fc rgb "#0000FF"
set object 54 rect from 0.101295, 40 to 64.772211, 50 fc rgb "#FFFF00"
set object 55 rect from 0.104321, 40 to 70.587637, 50 fc rgb "#FF00FF"
set object 56 rect from 0.113691, 40 to 78.754096, 50 fc rgb "#808080"
set object 57 rect from 0.126829, 40 to 79.323519, 50 fc rgb "#800080"
set object 58 rect from 0.127984, 40 to 79.744987, 50 fc rgb "#008080"
set object 59 rect from 0.128399, 40 to 147.195220, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.241735, 50 to 165.818249, 60 fc rgb "#FF0000"
set object 61 rect from 0.199534, 50 to 124.791379, 60 fc rgb "#00FF00"
set object 62 rect from 0.201074, 50 to 125.153169, 60 fc rgb "#0000FF"
set object 63 rect from 0.201437, 50 to 127.008760, 60 fc rgb "#FFFF00"
set object 64 rect from 0.204421, 50 to 132.080087, 60 fc rgb "#FF00FF"
set object 65 rect from 0.212596, 50 to 140.241573, 60 fc rgb "#808080"
set object 66 rect from 0.225722, 50 to 140.764370, 60 fc rgb "#800080"
set object 67 rect from 0.226812, 50 to 141.133001, 60 fc rgb "#008080"
set object 68 rect from 0.227148, 50 to 150.000049, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
