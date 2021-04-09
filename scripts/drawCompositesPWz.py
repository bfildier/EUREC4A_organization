""""""






if __name__ == '__main__':




    ##-- reference pw values
    pw_bins = np.linspace(30,70,21)
    pw_centers = np.convolve(pw_bins,[0.5,0.5], mode='valid')

    # dims
    n_pw = pw_centers.size
    # n_qrad = data_day.dims['launch_time']
    n_z = f.z.size

    ##-- conditional distribution
    qrad_on_pw = np.nan*np.zeros((n_pw,n_z))

    for i_pw in range(n_pw):
        
        pw_min = pw_bins[i_pw]
        pw_max = pw_bins[i_pw+1]

        inds_qrad = np.logical_and(f.pw > pw_min,f.pw <= pw_max)
        qrad_on_pw[i_pw] = np.nanmean(f.lw_peaks.qrad_lw_smooth[inds_qrad,:],axis=0)


    ##-- show
    cmap=plt.cm.viridis_r

    fig,ax = plt.subplots(figsize=(6,5))

    h = ax.imshow(qrad_on_pw.T,
            aspect=5,
            origin='lower',
            extent=[pw_bins[0],pw_bins[-1],f.z[0]/1000,f.z[-1]/1000],
            cmap=cmap)

    ax.set_xlabel('PW (mm)')
    ax.set_ylabel('z (km)')
    ax.set_title('Longwave cooling on %s\n during EUREC$^4$A'%date.strftime("%Y-%m-%d"))

    # colorbar
    # plt.colorbar(h)
    norm = matplotlib.colors.Normalize(vmin=np.nanmin(qrad_on_pw), vmax=np.nanmax(qrad_on_pw))
    cb = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),
                ax=ax,shrink=0.9,pad=0.06)
    cb.set_label(r'Longwave $Q_{rad}$ (K/day)')

    plt.savefig(os.path.join(figdir,'Qrad_LW_on_z_vs_PW_%s.pdf'%(date.strftime("%Y%m%d"))),bbox_inches='tight')