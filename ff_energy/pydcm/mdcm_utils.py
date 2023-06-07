import numpy as np
import dcm_fortran as dcm
print(dcm)

mdcm_ = dcm.dcm_fortran

def mdcm_set_up(scan_fesp, scan_fdns,
                # mdcm_cxyz="/home/unibas/boittier/fdcm_project/"
                #           "mdcms/methanol/10charges.xyz",
                # mdcm_clcl="/home/unibas/boittier/MDCM/examples/multi-conformer/"
                #           "5-charmm-files/10charges.dcm",
                mdcm_cxyz=None,
                mdcm_clcl=None,
                local_pos=None):

    mdcm_.dealloc_all()

    Nfiles = len(scan_fesp)
    # Nchars = int(np.max([
        # len(filename) for filelist in [scan_fesp, scan_fdns]
        # for filename in filelist]))
    Nchars = 125

    esplist = np.empty([Nfiles, Nchars], dtype='c')
    dnslist = np.empty([Nfiles, Nchars], dtype='c')

    for ifle in range(Nfiles):
        print("ESP: ", scan_fesp[ifle])
        print("DNS: ", scan_fdns[ifle])
        esplist[ifle] = "{0}:{1}\n".format(scan_fesp[ifle].rjust(len(scan_fesp[ifle]) - Nchars),
                                     Nchars)
        dnslist[ifle] = "{0}:{1}".format(scan_fdns[ifle].rjust(len(scan_fdns[ifle]) - Nchars),
                                     Nchars)
        print(esplist[ifle].T, "\n\n", dnslist[ifle].T)

    # Load cube files, read MDCM global and local files
    mdcm_.load_cube_files(Nfiles, Nchars, esplist, dnslist)
    mdcm_.load_clcl_file(mdcm_clcl)
    mdcm_.load_cxyz_file(mdcm_cxyz)

    # Get and set local MDCM array (to check if manipulation is possible)
    clcl = mdcm_.mdcm_clcl
    mdcm_.set_clcl(clcl)

    if local_pos is not None:
        mdcm_.set_clcl(local_pos)

    # Get and set global MDCM array (to check if manipulation is possible)
    cxyz = mdcm_.mdcm_cxyz
    mdcm_.set_cxyz(cxyz)

    # Write MDCM global from local and Fitted ESP cube files
    mdcm_.write_cxyz_files()
    mdcm_.write_mdcm_cube_files()

    return mdcm_
