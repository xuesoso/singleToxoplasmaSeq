from _loadlib.toxo_lib import *

# RH cell cycle (experiment 07)
htseq_input = [input_folder_dir + '180910_0701+0702_ns18_exon_htseq.tab.gz']
star_input = [input_folder_dir + '180910_0701+0702_ns18_exon_star.tab.gz']

output_dir = os.path.join(script_path, '../../Submission_analysis/Figures')+'/'
scv.settings.figdir = output_dir
sc.settings.figdir = output_dir
scv.settings._frameon = True
cell_cycle_cmap = ListedColormap(sns.color_palette(n_colors=5));
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
