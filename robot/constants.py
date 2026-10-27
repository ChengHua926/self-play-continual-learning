"""Numbers shared by the robosuite world (world.py), our copies of it (sim.py, gpu_world.py) and the
machine (robot_machine.py), kept here so that modules which do not need robosuite need not import it."""

# Control steps in one experiment: 2.5 s. Any situation can be a start, so an experiment need not be
# as long as a human demo (median 150 steps), and short ones give more different starts for the compute.
STEPS = 50
STATE_SIZE = 33
TABLE_TOP = 0.82            # height of the table's surface, in meters
