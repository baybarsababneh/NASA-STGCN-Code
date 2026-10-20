"""Terminate a Colab worker when its owning Python kernel disappears."""
import ctypes,os,signal,sys

def arm_parent_death_guard(expected_parent):
    if sys.platform!='linux':raise RuntimeError('Colab recovery worker requires the Linux parent-death guard')
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(1,signal.SIGTERM,0,0,0)!=0:
        raise OSError(ctypes.get_errno(),'Unable to arm PR_SET_PDEATHSIG; execution stopped')
    # Close the race between process creation and setting the death signal.
    if os.getppid()!=int(expected_parent):
        os.kill(os.getpid(),signal.SIGTERM)
        raise RuntimeError('Owning Colab kernel disappeared before worker initialization')

