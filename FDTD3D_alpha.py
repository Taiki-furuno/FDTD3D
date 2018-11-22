## /usr/bin/env python
# coding: utf-8

#   FDTD法で視る音の世界, コロナ社　の付録"fdtd_3d_c_sjis.c"をpythonにそのまま移植したコード
#  3次元でpmlを使ってる点が参考になる。
# 格子の設定方法が独特なことに注意
# 位置時刻前のvとpは保存しておかなくていいみたい
# px[0,:,:]とかvx[0,:,:]は常0???? 始点と終点がめちゃ曖昧。とりあえず0でない方式で回す。


# "fdtd_3d_c_sjis.c"からの変更点
#alpn = 0.0の場合は考えない。ifなどを省略
#メモリの確保とかのコードはない
#配列演算のforループはすべてnumpyのインデクシング？で一行で書くようにした
#時間のforループは必要
#ファイルの読み書きコードも省略
#変数名変更ix,jx,kx,tx ⇒ ix,jy,kz,tx




# In[1]:

import matplotlib.pyplot as plt
import numpy as np

# %%
# ■■■定数の宣言■■■

xmax = 5.000e0  # x軸解析領域 [m]
ymax = 5.000e0  # y軸解析領域 [m]
zmax = 5.000e0  # z軸解析領域 [m]
tmax = 2.000e-2	# 解析時間 [s]
dh = 5.000e-2	# 空間離散化幅 [m]
dt = 8.400e-5	# 時間離散化幅 [s]
c0 = 3.435e2		# 空気の音速 [m/s]
row0 = 1.205e0		# 空気の密度 [kg/m^3]
xdr = 2.000e0		# x軸音源位置 [m]
ydr = 3.000e0		# y軸音源位置 [m]
zdr = 2.500e0		# z軸音源位置 [m]
xon = 2.500e0		# 直方体x座標最小値 [m]
xox = 3.500e0		# 直方体x座標最大値 [m]
yon = 1.500e0		# 直方体y座標最小値 [m]
yox = 3.000e0		# 直方体y座標最大値 [m]
zon = 1.500e0		# 直方体z座標最小値 [m]
zox = 3.500e0		# 直方体z座標最大値 [m]
alpn = 0.200e0		# 直方体表面吸音率 [-]
m = 1.000e0		# ガウシアンパルス最大値 [m^3/s]
a = 2.000e6		# ガウシアンパルス係数 [-]
t0 = 3.000e-3	# ガウシアンパルス中心時間 [s]
pl = 16			# PML層数 [-]
pm = 4			# PML減衰係数テーパー乗数 [-]
emax = 1.200e0		# PML減衰係数最大値



# %%
# ■■■諸定数の算出■■■

# ■解析範囲■
ix = int(xmax / dh) + pl * 2
jy	= int(ymax / dh) + pl * 2
kz	= int(zmax / dh) + pl * 2  # kx	= int(xmax / dh) + pl * 2 だったけどさすがにkzだろう
tx	= int(tmax / dt)



# ■直方体位置■
ion	= int(xon / dh) + pl
iox	= int(xox / dh) + pl
jon	= int(yon / dh) + pl
jox = int(yox / dh) + pl
kon	= int(zon / dh) + pl
kox = int(zox / dh) + pl

# ■加振点位置■
idr	= int(xdr / dh) + pl
jdr	= int(ydr / dh) + pl
kdr	= int(zdr / dh) + pl

# ■加振時間■
tdr	= int((2.0 * t0) / dt)

# ■体積弾性率■
kp0	= row0 * c0 ** 2.0

# ■特性インピーダンス■
z0	= row0 * c0

# ■表面インピーダンス■  ?????

zn	= row0 * c0 * (1.0 + (1.0 - alpn)**0.5) / (1.0 - (1.0 - alpn**0.5))


# ■Courant数■
clf	= c0 * dt / dh


# ■粒子速度用更新係数■
vc	= clf / z0

# ■音圧用更新係数■
pc	= clf * z0


# ■PML用更新係数■
#!$omp parallel do private(i)
ex	= emax * ((np.arange(pl) / pl) ** pm)

#!$omp parallel do private(i)
pmla	= (1.0 - ex) / (1.0 + ex)
pmlb	= clf / z0 / (1.0 + ex)
pmlc	= clf * z0 / (1.0 + ex)

# %%
# ■■■変数の初期化■■■

#!$omp parallel do private(i, j, k)
p = np.zeros([ix, jy, kz])
px = np.zeros([ix, jy, kz])
py = np.zeros([ix, jy, kz])
pz = np.zeros([ix, jy, kz])
vx= np.zeros([ix, jy, kz])
vy= np.zeros([ix, jy, kz])
vz= np.zeros([ix, jy, kz])



#!$omp parallel do private(t)
q	= np.zeros([tdr])


# %%
# ■■■音源波形の作成■■■

#!$omp parallel do private(t)
q = m * np.exp(-a * (np.arange(tdr) * dt - t0) ** 2.0)







# %% 
# ■■■時間ループ■■■

tcount	= 1
fcount	= 0
txstep	= tx / 100.0
for t in range(tdr):
    print(t)
    
# %% 
    # ■粒子速度(vx)の更新■
#	!$omp parallel do private(i, j, k)
    vx[:pl,:,:]	= pmla.reshape(pl,1,1) * vx[:pl,:,:] - pmlb.reshape(pl,1,1) * (p[1:pl+1,:,:] - p[:pl,:,:])
    vx[pl:ix-pl-1,:,:]	= vx[pl:ix-pl-1,:,:] - vc * (p[pl+1:ix-pl,:,:] - p[pl:ix-pl-1,:,:]) 
    vx[ix-pl-1:ix-1,:,:]	= pmla[::-1].reshape(pl,1,1)  * vx[ix-pl-1:ix-1,:,:] - pmlb[::-1].reshape(pl,1,1)  * (p[ix-pl:ix,:,:] - p[ix-pl-1:ix-1,:,:])
    
    
    # ■粒子速度(vy)の更新■
    vy[:,:pl,:]	= pmla.reshape(1,pl,1) * vy[:,:pl,:] - pmlb.reshape(1,pl,1)  * (p[:,1:pl+1,:] - p[:,:pl,:])
    vy[:,pl:jy-pl-1,:]	= vy[:,pl:jy-pl-1,:] - vc * (p[:,pl+1:jy-pl,:] - p[:,pl:jy-pl-1,:]) 
    vy[:,jy-pl-1:jy-1,:]	= pmla[::-1].reshape(1,pl,1)  * vy[:,jy-pl-1:jy-1,:] - pmlb[::-1].reshape(1,pl,1)  * (p[:,jy-pl:jy,:] - p[:,jy-pl-1:jy-1,:])
    
    
    # ■粒子速度(vz)の更新■
    vz[:,:,:pl]	= pmla.reshape(1,1,pl) * vz[:,:,:pl] - pmlb.reshape(1,1,pl)  * (p[:,:,1:pl+1] - p[:,:,:pl])
    vz[:,:,pl:kz-pl-1]	= vz[:,:,pl:kz-pl-1] - vc * (p[:,:,pl+1:kz-pl] - p[:,:,pl:kz-pl-1]) 
    vz[:,:,kz-pl-1:kz-1]	= pmla[::-1].reshape(1,1,pl)  * vz[:,:,kz-pl-1:kz-1] - pmlb[::-1].reshape(1,1,pl)  * (p[:,:,kz-pl:kz] - p[:,:,kz-pl-1:kz-1])
    
    
    
    
    
    
# %%    
    # ■境界条件(vx)の計算■
#	!$omp parallel do private(j, k)
    vx[0,:,:]       = 0.0
    vx[ix-1,:,:]	= 0.0

    vx[ion-1,jon:jox,kon:kox] =  p[ion-1,jon:jox,kon:kox] / zn
    vx[iox,jon:jox,kon:kox]   = -p[iox+1,jon:jox,kon:kox] / zn


#	!■境界条件(vy)の計算■
#	!$omp parallel do private(k, i)
    vy[:,0,:]   	= 0.0
    vy[:,jy-1,:]	= 0.0  # jy-1とか書かずに-1で一番逆というのを指定したほうがミス少なそう
    
    vy[ion:iox,jon-1,kon:kox] =  p[ion:iox,jon-1,kon:kox] / zn
    vy[ion:iox,jox,kon:kox]   = -p[ion:iox,jox+1,kon:kox] / zn
    
#
#	!■境界条件(vz)の計算■
#	!$omp parallel do private(i, j)
    vz[:,:,0]       = 0.0
    vz[:,:,kz-1]	= 0.0
    
    vz[ion:iox,jon:jox,kon-1] =  p[ion:iox,jon:jox,kon-1] / zn
    vz[ion:iox,jon:jox,kox]   = -p[ion:iox,jon:jox,kox+1] / zn
#	!$omp parallel do private(i, j)

    
    
    # %% 
#    	!■音圧(px)の更新■
    px[:pl,:,:] = pmla.reshape(pl,1,1) * px[:pl,:,:] - pmlc.reshape(pl,1,1) * (vx[1:pl+1,:,:] - vx[:pl,:,:])  # a,b,cの使い分けはなに？  pの端は含むべきか否か、含まない気がする。

    px_at_source = px[idr,jdr,kdr]
    px[pl:ix-pl-1,:,:]	= px[pl:ix-pl-1,:,:] - pc * (px[pl:ix-pl-1,:,:] - px[pl-1:ix-pl-2,:,:])
    px[idr,jdr,kdr] = px_at_source + dt * kp0 * q[t]/3.0/(dh**3)
    
    px[ix-pl-1:ix-1,:,:]	= pmla[::-1].reshape(pl,1,1)  * px[ix-pl-1:ix-1,:,:] - pmlc[::-1].reshape(pl,1,1)  * (p[ix-pl:ix,:,:] - p[ix-pl-1:ix-1,:,:])
#
#	!■音圧(py)の更新■
    py[:,:pl,:]	= pmla.reshape(1,pl,1) * py[:,:pl,:] - pmlc.reshape(1,pl,1)  * (p[:,1:pl+1,:] - p[:,:pl,:])
    
    py_at_source = py[idr,jdr,kdr]
    py[:,pl:jy-pl-1,:]	= py[:,pl:jy-pl-1,:] - vc * (p[:,pl:jy-pl-1,:] - p[:,pl-1:jy-pl-2,:]) 
    py[idr,jdr,kdr] = py_at_source + dt * kp0 * q[t]/3.0/(dh**3)
    
    py[:,jy-pl-1:jy-1,:]	= pmla[::-1].reshape(1,pl,1)  * py[:,jy-pl-1:jy-1,:] - pmlc[::-1].reshape(1,pl,1)  * (p[:,jy-pl:jy,:] - p[:,jy-pl-1:jy-1,:])
    
    

    
    
#	!■音圧(pz)の更新■
    pz[:,:,:pl]	= pmla.reshape(1,1,pl) * pz[:,:,:pl] - pmlc.reshape(1,1,pl)  * (p[:,:,1:pl+1] - p[:,:,:pl])
    pz_at_source = pz[idr,jdr,kdr]
    pz[:,:,pl:kz-pl-1]	= pz[:,:,pl:kz-pl-1] - vc * (p[:,:,pl:kz-pl-1] - p[:,:,pl-1:kz-pl-2]) 
    pz[idr,jdr,kdr] = pz_at_source + dt * kp0 * q[t]/3.0/(dh**3)
    pz[:,:,kz-pl-1:kz-1]	= pmla[::-1].reshape(1,1,pl)  * pz[:,:,kz-pl-1:kz-1] - pmlc[::-1].reshape(1,1,pl)  * (p[:,:,kz-pl:kz] - p[:,:,kz-pl-1:kz-1])
    
    
    # %%
#    	!■音圧の合成■
#	!$omp parallel do private(i, j, k)
    p = px + py + pz
    
    if t > 10:
        break
    # %%
    # 結果の出力
    print(p.max())
#    plt.imshow(p[idr,:,:])
#    plt.contourf(p[idr,:,:])
    plt.subplot(311)
    plt.contour(p[idr,:,:])
    plt.subplot(312)
    plt.contour(p[:,jdr,:])
    plt.subplot(313)
    plt.contour(p[:,:,kdr])
    plt.show()

    
    
    