import ark
import sys

ark.Profiler(ark.Plan.from_file(sys.argv[1])).run(
    iter=10, profile_processor_groups=False
)
