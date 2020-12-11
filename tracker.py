import numpy as np
from scipy.constants import e, c
from functools import partial

from PySixHead.beamloss import STLAperture

try:
    from pycuda import cumath
    import pycuda.driver as drv
    context = drv.Context.get_current() #pycuda.autoinit.context

    from PySixHead.gpu_helpers import provide_pycuda_array, gpuarray_memcpy

    from PySixHead.beamloss import STLApertureGPU

    has_gpu = True
except Exception as e:
    print (f'GPU not available: {e}')
    has_gpu = False


def compute_rpp_psigma_rvv(pyht_beam, mathlib=np):
    '''Compute rpp, psigma and rvv from delta (pyht_beam.dp) and
    attach to pyht_beam. Uses given mathlib for computing sqrt.
    '''
    pyht_beam.rpp = 1. / (pyht_beam.dp + 1)

    restmass = pyht_beam.mass * c**2
    restmass_sq = restmass**2
    E0 = np.sqrt((pyht_beam.p0 * c)**2 + restmass_sq)
    p = pyht_beam.p0 * (1 + pyht_beam.dp)
    E = mathlib.sqrt((p * c) * (p * c) + restmass_sq)
    pyht_beam.psigma =  (E - E0) / (pyht_beam.beta * pyht_beam.p0 * c)

    gamma = E / restmass
    beta = mathlib.sqrt(1 - 1. / (gamma * gamma))
    pyht_beam.rvv = beta / pyht_beam.beta


def add_STL_attrs_to_PyHT_beam(pyht_beam, longitudinal_update=True):
    '''Upgrade PyHEADTAIL.Particles instance pyht_beam
    with all relevant attributes required by SixTrackLib.
    Thus, any reordering of pyht_beam (e.g. due to
    slicing or beam loss) will apply to the new
    attributes from SixTrackLib as well.

    Arguments:
        - pyht_beam: PyHEADTAIL beam instance
        - longitudinal_update: default True. False will speed up
                memory transfer from PyHEADTAIL to SixTrackLib by
                not recomputing the longitudinal rvv, rpp, psigma
                coordinates from delta [expert setting! Useful e.g. for
                transverse 2.5D PIC where only xp and yp are updated!]
    '''
    assert not any(map(
            lambda a: hasattr(pyht_beam, a),
            ['state', 'at_turn', 'at_element', 's',
             'has_STL_API', 'STL_longitudinal_update'])
        ), 'pyht_beam already has been updated with SixTrackLib attributes!'

    n = pyht_beam.macroparticlenumber

    coords_n_momenta_dict = {
        'state': np.ones(n, dtype=np.int64),
        'at_turn': np.zeros(n, dtype=np.int64),
        'at_element': np.zeros(n, dtype=np.int64),
        's': np.zeros(n, dtype=np.float64),
    }

    pyht_beam.update(coords_n_momenta_dict)
    pyht_beam.id = pyht_beam.id.astype(np.int64)

    pyht_beam.STL_longitudinal_update = longitudinal_update
    if not longitudinal_update:
        print ('\n\n*** Attention: longitudinal_update == False !!!\n'
               'That means any updates of delta (dp) outside of SixTrackLib\n'
               'are *TREATED WRONGLY*! The longitudinal momentum is assumed\n'
               'to stay constant in any kicks outside of SixTrackLib.\n'
               '(Otherwise set longitudinal_update to True when calling\n'
               'add_STL_attrs_to_PyHT_beam!)\n\n')
        compute_rpp_psigma_rvv(pyht_beam)
        pyht_beam.coords_n_momenta.update({'rpp', 'psigma', 'rvv'})

    pyht_beam.has_STL_API = True


class STLTracker(object):
    '''Tracker class for SixTrackLib through TrackJob interface.'''
    trackjob = None
    n_elements = 0

    def __init__(self, trackjob, i_start, i_end=None,
                 allow_losses=False):
        '''Tracker class for SixTrackLib through sixtracklib.TrackJob
        interface supporting openCL on multi-core CPU architectures.
        For CUDA / NVIDIA use the STLTrackerGPU class.

        Arguments:
            - trackjob: sixtracklib.TrackJob for openCL (on CPU) or plain CPU
            - i_start: index of element to start the track_line with
            - i_end: index of element until before which the track_line
                    should track. None or -1 will default to the last
                    element in the trackjob.beam_elements_buffer .
            - allow_losses: whether to transfer SixTrackLib lost particles
                    to PyHEADTAIL via PyHEADTAIL.aperture (this
                    will reorder the particles). Default False.
        '''
        if STLTracker.trackjob is None:
            STLTracker.trackjob = trackjob

        STLTracker.n_elements = trackjob.beam_elements_buffer.n_objects

        self.i_start = i_start
        if i_end and i_end != -1:
            self.i_end = i_end
        else:
            self.i_end = self.n_elements
        self.is_last_element = (self.i_end == self.n_elements)

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

    compute_rpp_psigma_rvv = staticmethod(compute_rpp_psigma_rvv)

    def pyht_to_stl(self, beam):
        stl_particles = self.trackjob.particles_buffer.get_object(0)
        n = beam.macroparticlenumber

        stl_particles.x[:n] = beam.x
        stl_particles.px[:n] = beam.xp
        stl_particles.y[:n] = beam.y
        stl_particles.py[:n] = beam.yp
        stl_particles.zeta[:n] = beam.z
        stl_particles.delta[:n] = beam.dp

        if not (hasattr(beam, 'has_STL_API') and beam.has_STL_API):
            add_STL_attrs_to_PyHT_beam(beam, longitudinal_update=True)

        stl_particles.particle_id[:n] = beam.id
        stl_particles.state[:n] = beam.state
        stl_particles.at_turn[:n] = beam.at_turn
        stl_particles.at_element[:n] = beam.at_element
        stl_particles.s[:n] = beam.s

        # further longitudinal coordinates of SixTrackLib
        if beam.STL_longitudinal_update:
            self.compute_rpp_psigma_rvv(beam)

        stl_particles.rpp[:n] = beam.rpp
        stl_particles.psigma[:n] = beam.psigma
        stl_particles.rvv[:n] = beam.rvv

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
        if not beam.STL_longitudinal_update:
            beam.rpp = stl_particles.rpp[pyht_visible]
            beam.psigma = stl_particles.psigma[pyht_visible]
            beam.rvv = stl_particles.rvv[pyht_visible]


if has_gpu:
    class STLTrackerGPU(object):
        '''Tracker class for SixTrackLib through CudaTrackJob.'''
        cudatrackjob = None
        pointers = {}
        context = None
        n_elements = 0

        def __init__(self, cudatrackjob, i_start, i_end, context=context,
                     allow_losses=False, longitudinal_update=True):
            '''Tracker class for SixTrackLib through sixtracklib.CudaTrackJob
            interface, requires NVIDIA GPU with CUDA.
            For openCL on CPU use STLTracker class.

            Arguments:
                - cudatrackjob: sixtracklib.CudaTrackJob instance
                - i_start: index of element to start the track_line with
                - i_end: index of element until before which the track_line
                        should track. None or -1 will default to the last
                        element in the trackjob.beam_elements_buffer
                - context: pycuda context object (see note below!)
                - allow_losses: whether to transfer SixTrackLib lost particles
                        to PyHEADTAIL via PyHEADTAIL.aperture (this
                        will reorder the particles). Default False.

            !!! Important: initialise sixtracklib.CudaTrackJob before
            initialising PyHEADTAIL or PyPIC objects, otherwise the contexts
            are not called in proper order and the interface does not work!
            (Typical errors relate to unreachable memory which has been
            allocated in alien contexts.)
            '''
            if STLTrackerGPU.cudatrackjob is None:
                STLTrackerGPU.cudatrackjob = cudatrackjob

                n_mp = int(
                    cudatrackjob.particles_buffer.get_object(0).num_particles)

                cudatrackjob.fetch_particle_addresses()
                assert cudatrackjob.last_status_success
                # particleset==0 is default:
                ptr = cudatrackjob.get_particle_addresses()

                STLTrackerGPU.pointers.update({
                    'x': provide_pycuda_array(ptr.contents.x, n_mp),
                    'px': provide_pycuda_array(ptr.contents.px, n_mp),
                    'y': provide_pycuda_array(ptr.contents.y, n_mp),
                    'py': provide_pycuda_array(ptr.contents.py, n_mp),
                    'z': provide_pycuda_array(ptr.contents.zeta, n_mp),
                    'delta': provide_pycuda_array(ptr.contents.delta, n_mp),
                    'rpp': provide_pycuda_array(ptr.contents.rpp, n_mp),
                    'psigma': provide_pycuda_array(ptr.contents.psigma, n_mp),
                    'rvv': provide_pycuda_array(ptr.contents.rvv, n_mp),
                    'id': provide_pycuda_array(
                        ptr.contents.particle_id, n_mp, dtype=np.int64),
                    'state': provide_pycuda_array(
                        ptr.contents.state, n_mp, dtype=np.int64),
                    'at_turn': provide_pycuda_array(
                        ptr.contents.at_turn, n_mp, dtype=np.int64),
                    'at_element': provide_pycuda_array(
                        ptr.contents.at_element, n_mp, dtype=np.int64),
                    's': provide_pycuda_array(ptr.contents.s, n_mp),
                })

                STLTrackerGPU.n_elements = len(
                    cudatrackjob.beam_elements_buffer.get_elements())

            self.i_start = i_start
            if i_end and i_end != -1:
                self.i_end = i_end
            else:
                self.i_end = self.n_elements
            self.is_last_element = (self.i_end == self.n_elements)

            self.context = context

            self.allow_losses = allow_losses
            if allow_losses:
                self.aperture = STLApertureGPU(self)

        def track(self, beam):
            # pass arrays and convert units
            self.pyht_to_stl(beam)
            # track in SixTrackLib
            self.cudatrackjob.track_line(self.i_start, self.i_end,
                                    finish_turn=self.is_last_element)
            # to be replaced by barrier:
            self.cudatrackjob.collectParticlesAddresses()

            assert self.cudatrackjob.last_track_status_success
            # pass arrays back (converting units back)
            self.stl_to_pyht(beam)

            # apply SixTrackLib beam loss to PyHEADTAIL:
            if self.allow_losses:
                self.transfer_losses(beam)

        def transfer_losses(self, beam):
                ### this reorders the particles arrays!
                self.aperture.track(beam)

        compute_rpp_psigma_rvv = staticmethod(partial(
            compute_rpp_psigma_rvv, mathlib=cumath))

        def pyht_to_stl(self, beam):
            self.memcpy(self.pointers['x'], beam.x)
            self.memcpy(self.pointers['px'], beam.xp)
            self.memcpy(self.pointers['y'], beam.y)
            self.memcpy(self.pointers['py'], beam.yp)
            self.memcpy(self.pointers['z'], beam.z)
            self.memcpy(self.pointers['delta'], beam.dp)

            if not (hasattr(beam, 'has_STL_API') and beam.has_STL_API):
                add_STL_attrs_to_PyHT_beam(beam, longitudinal_update=True)

            self.memcpy(self.pointers['id'], beam.id)
            self.memcpy(self.pointers['state'], beam.state)
            self.memcpy(self.pointers['at_turn'], beam.at_turn)
            self.memcpy(self.pointers['at_element'], beam.at_element)
            self.memcpy(self.pointers['s'], beam.s)

            # further longitudinal coordinates of SixTrackLib
            if beam.STL_longitudinal_update:
                self.compute_rpp_psigma_rvv(beam)

            self.memcpy(self.pointers['rpp'], beam.rpp)
            self.memcpy(self.pointers['psigma'], beam.psigma)
            self.memcpy(self.pointers['rvv'], beam.rvv)

            self.context.synchronize()

        memcpy = staticmethod(gpuarray_memcpy)

        def stl_to_pyht(self, beam):
            if self.allow_losses:
                pyht_visible = np.s_[:beam.macroparticlenumber]
            else:
                pyht_visible = np.s_[:]
            beam.x = self.pointers['x'][pyht_visible]
            beam.xp = self.pointers['px'][pyht_visible]
            beam.y = self.pointers['y'][pyht_visible]
            beam.yp = self.pointers['py'][pyht_visible]
            beam.z = self.pointers['z'][pyht_visible]
            beam.dp = self.pointers['delta'][pyht_visible]
            beam.id = self.pointers['id'][pyht_visible]
            beam.state = self.pointers['state'][pyht_visible]
            beam.at_turn = self.pointers['at_turn'][pyht_visible]
            beam.at_element = self.pointers['at_element'][pyht_visible]
            beam.s = self.pointers['s'][pyht_visible]
            if not beam.STL_longitudinal_update:
                beam.rpp = self.pointers['rpp'][pyht_visible]
                beam.psigma = self.pointers['psigma'][pyht_visible]
                beam.rvv = self.pointers['rvv'][pyht_visible]
