from _loadlib.toxo_lib import *

# ME49 Day 0 and Day 3 after induction (experiment 011)
htseq_input = [input_folder_dir + '190131_011_ns22_me49_union_exon.tab.gz']
star_input = [input_folder_dir + '190131_011_ns22_me49_star.tab.gz']

output_dir = os.path.join(script_path, '../../Submission_analysis/Figures')+'/'
scv.settings.figdir = output_dir
sc.settings.figdir = output_dir
