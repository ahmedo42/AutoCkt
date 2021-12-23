import os

import numpy as np
import scipy.interpolate as interp
import scipy.optimize as sciopt
from autockt.eval_engines.ngspice.ngspice_wrapper import NgSpiceWrapper

class CsAmpClass(NgSpiceWrapper):

    def translate_result(self, output_path):
        """
        :param output_path:
        :return
            result: dict(spec_kwds, spec_value)
        """

        # use parse output here
        freq, vout, Ibias = self.parse_output(output_path)
        bw = self.find_bw(vout, freq)
        gain = self.find_dc_gain(vout)


        spec = dict(
            bw=bw,
            gain=gain,
            Ibias=Ibias
        )

        return spec

    def parse_output(self, output_path):

        ac_fname = os.path.join(output_path, 'ac.csv')
        dc_fname = os.path.join(output_path, 'dc.csv')

        if not os.path.isfile(ac_fname) or not os.path.isfile(dc_fname):
            print("ac/dc file doesn't exist: %s" % output_path)

        ac_raw_outputs = np.genfromtxt(ac_fname, skip_header=1)
        dc_raw_outputs = np.genfromtxt(dc_fname, skip_header=1)
        freq = ac_raw_outputs[:, 0]
        vout = ac_raw_outputs[:, 1]
        ibias = -dc_raw_outputs[1]

        return freq, vout, ibias

    def find_dc_gain (self, vout):
        return np.abs(vout)[0]

    def find_bw(self, vout, freq):
        gain = np.abs(vout)
        gain_3dB = gain[0] / np.sqrt(2)
        return self._get_best_crossing(freq, gain, gain_3dB)


    def _get_best_crossing(self, xvec, yvec, val):
        interp_fun = interp.InterpolatedUnivariateSpline(xvec, yvec)

        def fzero(x):
            return interp_fun(x) - val

        xstart, xstop = xvec[0], xvec[-1]
        try:
            return sciopt.brentq(fzero, xstart, xstop)
        except ValueError:
            # avoid no solution
            if abs(fzero(xstart)) < abs(fzero(xstop)):
                return xstart
            return xstop