import numpy as np
from lsst.sims.maf.metrics import BaseMetric
from lsst.sims.maf.utils.astrometryUtils import m52snr, astrom_precision

class DCR_Precision(BaseMetric):
    """This metric trys to quantify how well the DCR effect can be contrained/measured."""
    
    def __init__(self, band, src_mag=22, seeingCol='seeingFwhmGeom', m5Col='fiveSigmaDepth',
                 PACol='paraAngle', filterCol='filter', atm_err=0.01, PA=False, **kwargs):
        
        self.band = band # required
        self.src_mag = src_mag
        self.m5Col = m5Col
        self.PACol = PACol
        self.seeingCol = seeingCol
        self.filterCol = filterCol
        self.atm_err = 0.01
        self.PA = PA
        self.metricName = f'DCR_{src_mag}_{self.band}'
        
        cols=['airmass', self.filterCol, self.m5Col, self.PACol, self.seeingCol]
        super(DCR_Precision, self).__init__(col=cols, metricName=self.metricName, **kwargs)
        
    def run(self, dataSlice, slicePoint=None):
        
        # get the data only corresponding the the desired filter
        data_filt = dataSlice[np.where(dataSlice[self.filterCol] == self.band)]
        
        # Compute the SNR from the observed mag and limiting mag
        # https://sims-maf.lsst.io/_modules/lsst/sims/maf/utils/astrometryUtils.html#m52snr
        snr = m52snr(self.src_mag, data_filt[self.m5Col])

        # The positional error is just the seeing scaled by the SNR with the error floor added in quadrature
        # https://sims-maf.lsst.io/_modules/lsst/sims/maf/utils/astrometryUtils.html#astrom_precisio
        pos_var = np.power(astrom_precision(data_filt[self.seeingCol], snr), 2) \
                  + self.atm_err**2
        pos_err = np.sqrt(pos_var)
        
        # compute tan(Z)
        zenith = np.arccos(1/data_filt['airmass'])
        
        # note that the Peter Yoachim's original code includes the PA 
        # https://github.com/lsst/sims_maf/blob/master/python/lsst/sims/maf/metrics/dcrMetric.py
        # however, it drops out in the end

        x_coord = np.tan(zenith)
        
        # function is of form, y=ax. a=y/x. da = dy/x.
        # Only strictly true if we know the unshifted position. But this should be a reasonable approx  
        slope_uncerts = pos_err/x_coord
        
        # error propagation
        # Assuming we know the unshfted position of the object (or there's little covariance if we are fitting for both)
        total_slope_uncert = 1./np.sqrt(np.sum(1./slope_uncerts**2))
        
        # Note that we no longer use the Bevington's method implemented by Lynne Jones at:
        # https://sims-maf.lsst.io/lsst.sims.maf.utils.html
        # as it assumes the line has a form of y=a+bx and our intercept is simply zero.
        
        result = total_slope_uncert
        return result