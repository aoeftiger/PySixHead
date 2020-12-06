from PySixHead.tracker import STLTracker, add_STL_attrs_to_PyHT_beam

# GPU:
try:
    from PySixHead.tracker import STLTrackerGPU
except ImportError:
    pass

"""
                                      &&&@
                             &&&      @&&&&
                            &&&&&      &&&&&          &
                             &&&&        &&&       &&&&
                  &&@         &&&        *&&#    @&&&&@       &&&&&
                  @&&&&&        &&&      &&&    @&&&       @&&&&&
            &&&     &&&&&        @&&&  &&&&    &&&        &&&&&   @&
           &&&&        &&&           &&&&     (&&       &&&&     @&&&
          @&&&&         &&&       &&&&  &&              &&&      &&&&&
           &&@        &&&&&&&&&&  &     &&&@ &&&&  &&&&&&&&&&     &&&
           &&     &&&&&       (&&&@ @&&&@ &&&   &&&&        &&&&  &&&
           &&&  &&&&  &&&   &&&  &&& &      &&&&&&&@     @&&@  &#/&&
            &&& *    &&&    @&&         @  &&@    @&&@@&   &&&   &&&
          &&  &&&& &&&* &&&& &&&&&&&& (& &&&    &&&&&&&&&&& &&& @&@ &&&&&&&&&@
   &&&&&&&&       @&& %@   &&&/    &&&&&   &&&&&       %   & &&&        @&&&&&&@
*&&&&&&           &&&     &&  &&&&&&           &&&&&   ,&&   &&&(
             &&& @ &&&    @&&&&%                   &&&&&    &&&&  &&&
         @&&&&   &&  && &&&&                           &&&@ && &&   &&&&&
    @&&&&&&      @&&   &&& ,    .d$$p.       _.g$$b.  &  &&&  &&&       &&&&&&
  &&&&&&@         &&& @& &&&         "T$p.   "     -.   (&&  &  &&&         &&&&&
                && &&&  &&&    . (@ )          (@ ) .  &&&&  &&& &&          &&&
              &&&   &*(&&&                             @ &&& &&  &&&(
            &&&&      &&& &&&           .  .         &&&*(&&&/     &&&
          &&&&&       &&&   &&           ""         &&  * &&#      @&&&&
          &&&&        &&& &@  @        __  __      &   &@ &&        &&&&&
          &&          &&& &&&@       .'  ""  '.    &&&  &&          &&&
                      @&&   &&&&&     ".____."   &&&&&   @&&
                      &&&    &&&&&&            @&&&&&    @&&
                     &&&&       @&&&          @&&&        &&&&
                    &&&&&                                 &&&&&
                   @&&&                                     &&&
"""
