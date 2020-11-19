import numpy as np
from scipy.constants import e, c

from PySixHead.beamloss import STLAperture

try:
    from pycuda import cumath
    import pycuda.gpuarray as gp
    from pycuda.driver import memcpy_dtod_async
    import pycuda.driver as drv

    context = drv.Context.get_current() #pycuda.autoinit.context

    def provide_pycuda_array(ptr, n_entries):
        return gp.GPUArray(n_entries, dtype=np.float64, gpudata=ptr)


    def gpuarray_memcpy(dest, src):
        '''Device memory copy with pycuda from
        src GPUArray to dest GPUArray.
        '''
    #     dest[:] = src
    #     memcpy_atoa(dest, 0, src, 0, len(src))
        memcpy_dtod_async(dest.gpudata, src.gpudata, src.nbytes)

    has_gpu = True
except Exception:
    has_gpu = False


class STLTracker(object):
    '''Tracker class for SixTrackLib through TrackJob interface.'''
    trackjob = None
    n_elements = 0
    mathlib = np

    def __init__(self, trackjob, i_start, i_end=None, allow_losses=False):
        '''Tracker class for SixTrackLib through sixtracklib.TrackJob
        interface. For CUDA / NVIDIA use the STLTrackerGPU class.

        Arguments:
            - trackjob: sixtracklib.TrackJob e.g. for openCL or cpu
            - i_start: index of element to start the track_line with
            - i_end: index of element until before which the track_line
                     should track. None or -1 will default to the last
                     element in the trackjob.beam_elements_buffer .
        '''
        if STLTracker.trackjob is None:
            STLTracker.trackjob = trackjob

        STLTracker.n_elements = trackjob.beam_elements_buffer.n_objects

        self.i_start = i_start
        if i_end and i_end != -1:
            self.i_end = i_end
        else:
            self.i_end = self.n_elements
        self.is_last_element = (i_end == self.n_elements)

        self.allow_losses = allow_losses
        if allow_losses:
            self.aperture = STLAperture(self)

    def track(self, beam):
        # pass arrays and convert units
        self.pyht_to_stl(beam)

        # track in SixTrackLib
        self.trackjob.track_line(self.i_start, self.i_end,
                            finish_turn=self.is_last_element)

        # pass arrays back (converting units back)
        self.stl_to_pyht(beam)

        # apply SixTrackLib beam loss to PyHEADTAIL:
        if self.allow_losses:
            self.transfer_losses(beam)

    def transfer_losses(self, beam):
        ### this reorders the particles arrays!
        self.aperture.track(beam)

    def pyht_to_stl(self, beam):
        stl_particles = self.trackjob.particles_buffer.get_object(0)
        n = beam.macroparticlenumber

        stl_particles.x[:n] = beam.x
        stl_particles.px[:n] = beam.xp
        stl_particles.y[:n] = beam.y
        stl_particles.py[:n] = beam.yp
        stl_particles.zeta[:n] = beam.z
        stl_particles.delta[:n] = beam.dp

        stl_particles.particle_id[:n] = beam.id
        stl_particles.state[:n] = beam.state
        stl_particles.at_turn[:n] = beam.at_turn
        stl_particles.at_element[:n] = beam.at_element
        stl_particles.s[:n] = beam.s

        # further longitudinal coordinates of SixTrackLib
        rpp = 1. / (beam.dp + 1)
        stl_particles.rpp[:n] = rpp

        restmass = beam.mass * c**2
        restmass_sq = restmass**2
        E0 = np.sqrt((beam.p0 * c)**2 + restmass_sq)
        p = beam.p0 * (1 + beam.dp)
        E = self.mathlib.sqrt((p * c) * (p * c) + restmass_sq)
        psigma =  (E - E0) / (beam.beta * beam.p0 * c)
        stl_particles.psigma[:n] = psigma

        gamma = E / restmass
        beta = self.mathlib.sqrt(1 - 1. / (gamma * gamma))
        rvv = beta / beam.beta
        stl_particles.rvv[:n] = rvv

        self.trackjob.push_particles()

    def stl_to_pyht(self, beam):
        if self.allow_losses:
            pyht_visible = np.s_[:beam.macroparticlenumber]
        else:
            pyht_visible = np.s_[:]

        self.trackjob.collect_particles()
        stl_particles = self.trackjob.particles_buffer.get_object(0)

        beam.x = stl_particles.x[pyht_visible]
        beam.xp = stl_particles.px[pyht_visible]
        beam.y = stl_particles.y[pyht_visible]
        beam.yp = stl_particles.py[pyht_visible]
        beam.z = stl_particles.zeta[pyht_visible]
        beam.dp = stl_particles.delta[pyht_visible]
        beam.id = stl_particles.particle_id[pyht_visible]
        beam.state = stl_particles.state[pyht_visible]
        beam.at_turn = stl_particles.at_turn[pyht_visible]
        beam.at_element = stl_particles.at_element[pyht_visible]
        beam.s = stl_particles.s[pyht_visible]


if has_gpu:
    class STLTrackerGPU(object):
        '''Tracker class for SixTrackLib through CudaTrackJob.'''
        cudatrackjob = None
        pointers = {}
        context = None
        n_elements = 0

        def __init__(self, cudatrackjob, i_start, i_end, context=context,
                     allow_losses=False):
            if STLTrackerGPU.cudatrackjob is None:
                STLTrackerGPU.cudatrackjob = cudatrackjob

                cudatrackjob.fetch_particle_addresses()
                assert cudatrackjob.last_status_success
                ptr = cudatrackjob.get_particle_addresses() # particleset==0 is default

                STLTrackerGPU.pointers.update({
                    'x': provide_pycuda_array(ptr.contents.x),
                    'px': provide_pycuda_array(ptr.contents.px),
                    'y': provide_pycuda_array(ptr.contents.y),
                    'py': provide_pycuda_array(ptr.contents.py),
                    'z': provide_pycuda_array(ptr.contents.zeta),
                    'delta': provide_pycuda_array(ptr.contents.delta),
                    'rpp': provide_pycuda_array(ptr.contents.rpp),
                    'psigma': provide_pycuda_array(ptr.contents.psigma),
                    'rvv': provide_pycuda_array(ptr.contents.rvv),
                    'id': provide_pycuda_array(ptr.contents.particle_id, np.int64),
                    'state': provide_pycuda_array(ptr.contents.state, np.int64),
                    'at_turn': provide_pycuda_array(ptr.contents.at_turn, np.int64),
                    'at_element': provide_pycuda_array(ptr.contents.at_element, np.int64),
                    's': provide_pycuda_array(ptr.contents.s, np.float64),
                })
                STLTrackerGPU.n_elements = len(
                    cudatrackjob.beam_elements_buffer.get_elements())

            self.i_start = i_start
            self.i_end = i_end
            self.is_last_element = (i_end == self.n_elements)

            self.context = context

            self.allow_losses = allow_losses
            if allow_losses:
                from PySixHead.beamloss import STLApertureGPU

                self.aperture = STLApertureGPU(self)

        def track(self, beam):
            # pass arrays and convert units
            self.pyht_to_stl(beam)
            # track in SixTrackLib
            cudatrackjob.track_line(self.i_start, self.i_end,
                                    finish_turn=self.is_last_element)
            # to be replaced by barrier:
            cudatrackjob.collectParticlesAddresses()

            assert cudatrackjob.last_track_status_success
            # pass arrays back (converting units back)
            self.stl_to_pyht(beam)

            # apply SixTrackLib beam loss to PyHEADTAIL:
            if self.allow_losses:
                self.transfer_losses(beam)

        def transfer_losses(self, beam):
                ### this reorders the particles arrays!
                self.aperture.track(beam)

        def pyht_to_stl(self, beam):
            self.memcpy(self.pointers['x'], beam.x)
            self.memcpy(self.pointers['px'], beam.xp)
            self.memcpy(self.pointers['y'], beam.y)
            self.memcpy(self.pointers['py'], beam.yp)
            self.memcpy(self.pointers['z'], beam.z)
            self.memcpy(self.pointers['delta'], beam.dp)

            self.memcpy(self.pointers['id'], beam.id)
            self.memcpy(self.pointers['state'], beam.state)
            self.memcpy(self.pointers['at_turn'], beam.at_turn)
            self.memcpy(self.pointers['at_element'], beam.at_element)
            self.memcpy(self.pointers['s'], beam.s)

            # further longitudinal coordinates of SixTrackLib
            rpp = 1. / (beam.dp + 1)
            self.memcpy(self.pointers['rpp'], rpp)

            restmass = beam.mass * c**2
            restmass_sq = restmass**2
            E0 = np.sqrt((beam.p0 * c)**2 + restmass_sq)
            p = beam.p0 * (1 + beam.dp)
            E = cumath.sqrt((p * c) * (p * c) + restmass_sq)
            psigma =  (E - E0) / (beam.beta * beam.p0 * c)
            self.memcpy(self.pointers['psigma'], psigma)

            gamma = E / restmass
            beta = cumath.sqrt(1 - 1. / (gamma * gamma))
            rvv = beta / beam.beta
            self.memcpy(self.pointers['rvv'], rvv)

            self.context.synchronize()

        memcpy = staticmethod(gpuarray_memcpy)

        def stl_to_pyht(self, beam):
            if self.allow_losses:
                all = np.s_[:beam.macroparticlenumber]
            else:
                all = np.s_[:]
            beam.x = self.pointers['x'][all]
            beam.xp = self.pointers['px'][all]
            beam.y = self.pointers['y'][all]
            beam.yp = self.pointers['py'][all]
            beam.z = self.pointers['z'][all]
            beam.dp = self.pointers['delta'][all]
            beam.id = self.pointers['id'][all]
            beam.state = self.pointers['state'][all]
            beam.at_turn = self.pointers['at_turn'][all]
            beam.at_element = self.pointers['at_element'][all]
            beam.s = self.pointers['s'][all]
