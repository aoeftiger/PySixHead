import numpy as np

from PyHEADTAIL.aperture.aperture import Aperture
from PyHEADTAIL.general import pmath as pm

try:
    from PySixHead.gpu_helpers import gpuarray_memcpy
    has_gpu = True
except Exception as e:
    print (f'GPU not available: {e}')
    has_gpu = False


class STLAperture(Aperture):
    '''Removes particles in PyHEADTAIL which have
    been lost in a SixTrackLib aperture (i.e. have state=0).
    '''
    def __init__(self, stl_tracker, *args, **kwargs):
        '''stl_tracker is a STLTracker instance with pointers
        pointing to the SixTrackLib.Particles attributes.
        These have length total number of macro-particles
        as initially started.
        In PyHEADTAIL the beam.macroparticlenumber will decrease
        with lost particles while in SixTrackLib the attribute arrays
        remain at the same original length and just state gets
        switched to 0 for each lost particle.
        '''
        self.pyht_to_stl = stl_tracker.pyht_to_stl
        self.stl_trackjob = stl_tracker.trackjob

    def relocate_lost_particles(self, beam, alive):
        '''Overwriting the Aperture.relocate_lost_particles
        in order to update the SixTrackLib arrays with the fully
        reordered PyHEADTAIL macro-particle arrays before
        they get cut to the decreased length of still
        alive macro-particles.
        '''
        # descending sort to have alive particles (the 1 entries) in the front
        perm = pm.argsort(-alive)

        beam.reorder(perm)

        n_alive = pm.sum(alive)
        # on CPU: (even if pm.device == 'GPU', as pm.sum returns np.ndarray)
        n_alive = np.int32(n_alive)

        ### additional part for SixTrackLib:
        self.pyht_to_stl(beam)
        ### also need to limit view on SixTrackLib attributes
        ### in PyHT beam for their next reordering
        beam.state = beam.state[:n_alive]
        beam.at_element = beam.at_element[:n_alive]
        beam.at_turn = beam.at_turn[:n_alive]
        beam.s = beam.s[:n_alive]
        if not beam.STL_longitudinal_update:
            beam.rpp = beam.rpp[:n_alive]
            beam.psigma = beam.psigma[:n_alive]
            beam.rvv = beam.rvv[:n_alive]

        return n_alive

    def tag_lost_particles(self, beam):
        '''Return mask of length beam.macroparticlenumber with
        alive particles being 1 and lost particles being 0.
        '''
        self.stl_trackjob.collect_particles()
        stl_particles = self.stl_trackjob.particles_buffer.get_object(0)
        alive = stl_particles.state[:beam.macroparticlenumber]
        return alive


if has_gpu:
    class STLApertureGPU(Aperture):
        '''Removes particles in PyHEADTAIL which have
        been lost in a SixTrackLib aperture (i.e. have state=0).
        '''
        def __init__(self, stl_tracker_gpu, *args, **kwargs):
            '''stl_tracker_gpu is a STLTrackerGPU instance with pointers
            pointing to the SixTrackLib.Particles attributes.
            These have length total number of macro-particles
            as initially started.
            In PyHEADTAIL the beam.macroparticlenumber will decrease
            with lost particles while in SixTrackLib the attribute arrays
            remain at the same original length and just state gets
            switched to 0 for each lost particle.
            '''
            self.pyht_to_stl = stl_tracker_gpu.pyht_to_stl
            self.stl_p = stl_tracker_gpu.pointers

        memcpy = staticmethod(gpuarray_memcpy)

        def relocate_lost_particles(self, beam, alive):
            '''Overwriting the Aperture.relocate_lost_particles
            in order to update the SixTrackLib arrays with the fully
            reordered PyHEADTAIL macro-particle arrays before
            they get cut to the decreased length of still
            alive macro-particles.
            '''
            # descending sort to have alive particles (the 1 entries) in the front
            perm = pm.argsort(-alive)

            beam.reorder(perm)

            n_alive = pm.sum(alive)
            # on CPU: (even if pm.device == 'GPU', as pm.sum returns np.ndarray)
            n_alive = np.int32(n_alive)

            ### additional part for SixTrackLib:
            self.pyht_to_stl(beam)
            self.memcpy(self.stl_p['state'], beam.state)
            self.memcpy(self.stl_p['at_turn'], beam.at_turn)
            self.memcpy(self.stl_p['at_element'], beam.at_element)
            self.memcpy(self.stl_p['s'], beam.s)
            if not beam.STL_longitudinal_update:
                self.memcpy(self.stl_p['rpp'], beam.rpp)
                self.memcpy(self.stl_p['psigma'], beam.psigma)
                self.memcpy(self.stl_p['rvv'], beam.rvv)
            ### also need to limit view on SixTrackLib attributes
            ### in PyHT beam for their next reordering
            beam.state = beam.state[:n_alive]
            beam.at_element = beam.at_element[:n_alive]
            beam.at_turn = beam.at_turn[:n_alive]
            beam.s = beam.s[:n_alive]
            if not beam.STL_longitudinal_update:
                beam.rpp = beam.rpp[:n_alive]
                beam.psigma = beam.psigma[:n_alive]
                beam.rvv = beam.rvv[:n_alive]

            return n_alive

        def tag_lost_particles(self, beam):
            '''Return mask of length beam.macroparticlenumber with
            alive particles being 1 and lost particles being 0.
            '''
            return self.stl_p['state'][:beam.macroparticlenumber]
