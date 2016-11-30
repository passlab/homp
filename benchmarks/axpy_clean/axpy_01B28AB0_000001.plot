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

set object 15 rect from 0.179637, 0 to 152.615316, 10 fc rgb "#FF0000"
set object 16 rect from 0.131541, 0 to 111.622212, 10 fc rgb "#00FF00"
set object 17 rect from 0.133833, 0 to 112.223451, 10 fc rgb "#0000FF"
set object 18 rect from 0.134333, 0 to 112.886543, 10 fc rgb "#FFFF00"
set object 19 rect from 0.135112, 0 to 113.776265, 10 fc rgb "#FF00FF"
set object 20 rect from 0.136185, 0 to 114.153367, 10 fc rgb "#808080"
set object 21 rect from 0.136642, 0 to 114.626662, 10 fc rgb "#800080"
set object 22 rect from 0.137444, 0 to 115.068209, 10 fc rgb "#008080"
set object 23 rect from 0.137725, 0 to 149.641020, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.177886, 10 to 151.424651, 20 fc rgb "#FF0000"
set object 25 rect from 0.114276, 10 to 97.350723, 20 fc rgb "#00FF00"
set object 26 rect from 0.116814, 10 to 98.018850, 20 fc rgb "#0000FF"
set object 27 rect from 0.117349, 10 to 98.739659, 20 fc rgb "#FFFF00"
set object 28 rect from 0.118246, 10 to 99.821719, 20 fc rgb "#FF00FF"
set object 29 rect from 0.119523, 10 to 100.229772, 20 fc rgb "#808080"
set object 30 rect from 0.120030, 10 to 100.771650, 20 fc rgb "#800080"
set object 31 rect from 0.120913, 10 to 101.283374, 20 fc rgb "#008080"
set object 32 rect from 0.121270, 10 to 148.188537, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.176128, 20 to 149.860921, 30 fc rgb "#FF0000"
set object 34 rect from 0.098273, 20 to 84.044218, 30 fc rgb "#00FF00"
set object 35 rect from 0.100864, 20 to 84.610318, 30 fc rgb "#0000FF"
set object 36 rect from 0.101295, 20 to 85.307752, 30 fc rgb "#FFFF00"
set object 37 rect from 0.102131, 20 to 86.323722, 30 fc rgb "#FF00FF"
set object 38 rect from 0.103369, 20 to 86.729283, 30 fc rgb "#808080"
set object 39 rect from 0.103831, 20 to 87.210105, 30 fc rgb "#800080"
set object 40 rect from 0.104663, 20 to 87.762001, 30 fc rgb "#008080"
set object 41 rect from 0.105109, 20 to 146.736851, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.174321, 30 to 150.965560, 40 fc rgb "#FF0000"
set object 43 rect from 0.075885, 30 to 66.014026, 40 fc rgb "#00FF00"
set object 44 rect from 0.079562, 30 to 67.211470, 40 fc rgb "#0000FF"
set object 45 rect from 0.080547, 30 to 69.279397, 40 fc rgb "#FFFF00"
set object 46 rect from 0.083105, 30 to 70.876521, 40 fc rgb "#FF00FF"
set object 47 rect from 0.084887, 30 to 71.444316, 40 fc rgb "#808080"
set object 48 rect from 0.085592, 30 to 72.209434, 40 fc rgb "#800080"
set object 49 rect from 0.086897, 30 to 73.158518, 40 fc rgb "#008080"
set object 50 rect from 0.087627, 30 to 144.904774, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.181208, 40 to 153.944990, 50 fc rgb "#FF0000"
set object 52 rect from 0.145563, 40 to 123.249605, 50 fc rgb "#00FF00"
set object 53 rect from 0.147759, 40 to 123.778075, 50 fc rgb "#0000FF"
set object 54 rect from 0.148137, 40 to 124.474661, 50 fc rgb "#FFFF00"
set object 55 rect from 0.148973, 40 to 125.449662, 50 fc rgb "#FF00FF"
set object 56 rect from 0.150151, 40 to 125.810066, 50 fc rgb "#808080"
set object 57 rect from 0.150583, 40 to 126.286701, 50 fc rgb "#800080"
set object 58 rect from 0.151389, 40 to 126.805153, 50 fc rgb "#008080"
set object 59 rect from 0.151778, 40 to 151.062552, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.182700, 50 to 155.032783, 60 fc rgb "#FF0000"
set object 61 rect from 0.160725, 50 to 135.941514, 60 fc rgb "#00FF00"
set object 62 rect from 0.162938, 50 to 136.449898, 60 fc rgb "#0000FF"
set object 63 rect from 0.163290, 50 to 137.124703, 60 fc rgb "#FFFF00"
set object 64 rect from 0.164104, 50 to 138.023646, 60 fc rgb "#FF00FF"
set object 65 rect from 0.165169, 50 to 138.327181, 60 fc rgb "#808080"
set object 66 rect from 0.165538, 50 to 138.791256, 60 fc rgb "#800080"
set object 67 rect from 0.166357, 50 to 139.288824, 60 fc rgb "#008080"
set object 68 rect from 0.166693, 50 to 152.376226, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
