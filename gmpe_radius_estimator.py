import numpy as np
from openquake.hazardlib.imt import PGA
from openquake.hazardlib.geo import Point
from openquake.hazardlib.gsim.boore_2014 import BooreEtAl2014
from openquake.hazardlib.contexts import SitesContext, RuptureContext, DistancesContext

def estimate_radius_pga_gmpe(mag, lat, lon, target_pga=0.05):
    """
    Estimate radius around the epicenter where PGA exceeds a threshold using BooreEtAl2014 GMPE.
    """

    gsim = BooreEtAl2014()
    imt = PGA()
    distances_km = np.linspace(1, 300, 100)

    for R in distances_km:
        # Build site context
        sites_ctx = SitesContext()
        sites_ctx.vs30 = np.array([760.0])
        sites_ctx.vs30measured = np.array([False])
        sites_ctx.backarc = np.array([False])
        sites_ctx.sids = np.array([0])  # Required by OpenQuake for indexing

        # Build rupture context
        rup_ctx = RuptureContext()
        rup_ctx.mag = mag
        rup_ctx.rake = 0.0
        rup_ctx.ztor = 0.0
        rup_ctx.dip = 90.0
        rup_ctx.width = 10.0
        rup_ctx.hypo_depth = 10.0
        rup_ctx.rrup = np.array([R])
        rup_ctx.rjb = np.array([R])
        rup_ctx.rx = np.array([R])
        rup_ctx.ry0 = np.array([R])
        rup_ctx.ry = np.array([R])
        rup_ctx.rvolc = np.array([R])

        # Build distances context
        dist_ctx = DistancesContext()
        dist_ctx.rrup = np.array([R])
        dist_ctx.rjb = np.array([R])
        dist_ctx.rx = np.array([R])
        dist_ctx.ry0 = np.array([R])
        dist_ctx.ry = np.array([R])
        dist_ctx.rvolc = np.array([R])

        # Compute PGA
        mean_ln, _ = gsim.get_mean_and_stddevs(
            sites_ctx, rup_ctx, dist_ctx, imt, stddev_types=[]
        )
        pga = np.exp(mean_ln[0])

        if pga < target_pga:
            return round(R, 2)

    return 300.0  # default max radius
