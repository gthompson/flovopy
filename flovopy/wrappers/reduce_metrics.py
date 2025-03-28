from obspy import UTCDateTime, read_inventory
from flovopy.core.sam import VSAM, VSEM, DSAM


def reduce_to_1km(paths, year, do_VR=False, do_VRS=False, do_ER=True, do_DR=True, do_DRS=True,
                  sampling_interval=60, invfile=None, source=None, Q=None, ext='pickle'):
    """
    Reduces VSAM, VSEM, and DSAM metrics to 1km source-referenced values (VR, ER, DR).

    Parameters:
        paths (dict): Dictionary of directory paths.
        year (int): Year of interest.
        do_VR (bool): Compute VR.
        do_VRS (bool): Compute VRS.
        do_ER (bool): Compute ER.
        do_DR (bool): Compute DR.
        do_DRS (bool): Compute DRS.
        sampling_interval (float): Sampling interval in seconds.
        invfile (str): Path to StationXML inventory file.
        source (tuple): Source location (lat, lon, elevation).
        Q (float or None): Attenuation quality factor.
        ext (str): File extension to use for reading/writing metrics.
    """
    startTime = UTCDateTime(year, 1, 1)
    endTime = UTCDateTime(year, 12, 31, 23, 59, 59.9)

    if not invfile or not os.path.isfile(invfile):
        print("Inventory file not found.")
        return
    if not source:
        print("Source location not provided.")
        return

    inv = read_inventory(invfile)

    if do_VR or do_VRS:
        vsam = VSAM.read(startTime, endTime, SAM_DIR=paths['SAM_DIR'], sampling_interval=sampling_interval, ext=ext)
        if do_VR:
            VRobj = vsam.compute_reduced_velocity(inv, source, surfaceWaves=False, Q=Q)
            VRobj.write(SAM_DIR=paths['SAM_DIR'], overwrite=True)
        if do_VRS:
            VRSobj = vsam.compute_reduced_velocity(inv, source, surfaceWaves=True, Q=Q)
            VRSobj.write(SAM_DIR=paths['SAM_DIR'], overwrite=True)

    if do_ER:
        vsem = VSEM.read(startTime, endTime, SAM_DIR=paths['SAM_DIR'], sampling_interval=sampling_interval, ext=ext)
        ERobj = vsem.compute_reduced_energy(inv, source, Q=Q)
        ERobj.write(SAM_DIR=paths['SAM_DIR'], overwrite=True)

    if do_DR or do_DRS:
        dsam = DSAM.read(startTime, endTime, SAM_DIR=paths['SAM_DIR'], sampling_interval=sampling_interval, ext=ext)
        if do_DR:
            DRobj = dsam.compute_reduced_displacement(inv, source, surfaceWaves=False, Q=Q)
            DRobj.write(SAM_DIR=paths['SAM_DIR'], overwrite=True)
        if do_DRS:
            DRSobj = dsam.compute_reduced_displacement(inv, source, surfaceWaves=True, Q=Q)
            DRSobj.write(SAM_DIR=paths['SAM_DIR'], overwrite=True)
